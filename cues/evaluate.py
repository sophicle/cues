"""Sample rollouts for one model under one prompt on one benchmark, with and without cues (vLLM).

    python -m cues eval --model olmo7b --benchmark math500 --cues none auto
    python -m cues eval --model qwen14b --prompt boxed --benchmark aime2025 --cues none $' Alright,' --n-rollouts 32
    python -m cues eval --model olmo7b --adapter $RUNS/rl/olmo7b/checkpoint-300 --label olmo7b_ck300 --max-new-tokens 7168

The protocol (paper App. B): temperature 0.6, top-p 0.95, a 31,744-token budget (7,168 for 8,192-context
checkpoints), 32 rollouts per problem taken as two batches of 16 (the second with --rollout-index-offset 16
--suffix _b2), the prompt's stop strings, and the same grader for every model. vLLM seeds sample i of a
request with seed + i, so the sampling seed here is --seed + --rollout-index-offset: rollout index r always
gets seed + r, and a second batch continues the first instead of repeating it (a batch seeded seed + 1
reproduces all but one of the first batch's samples). scripts/eval_grid.sh runs the full protocol sharded over a node's GPUs.

The cue is appended to the rendered prompt as raw text and stored as the head of the completion, so the
grader and the reader see the whole solution. The budget rule: max_tokens = min(requested + 1024, the
checkpoint's context) - 1024, recorded per record as max_tokens_used.

Output: <out>/<label>/<prompt>/<benchmark>/<cue slug>/rollouts_p<start>[suffix].jsonl, one record per
rollout: problem_key, rollout_index, cue, label, committed, correct, hit_token_cap, finish_reason,
output_tokens, max_tokens_used, completion. A file is skipped only when its configuration and rollout identities match. Legacy or mismatched
results require a new output root or label. Files are written
to a temp name then renamed, so an interrupted task never leaves a half file where a complete one was.

Provenance: sophicle/reason registers/evaluate.py.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cues import models as M
from cues.grading import is_correct
from cues.prompts import GRADING, PROMPTS, STOPS, render

DATA = Path(__file__).resolve().parent.parent / "data" / "problems"
RUNS = Path(os.environ.get("RUNS", "runs"))
BENCHMARKS = sorted(p.stem for p in DATA.glob("*.jsonl"))


def read_jsonl(path):
    """Completions carry U+2028/U+2029; str.splitlines() would shred records, so split on \\n only."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.read().split("\n") if line.strip()]


def problems_path(benchmark):
    p = Path(benchmark)
    if p.suffix == ".jsonl" or "/" in benchmark:
        return p
    return DATA / f"{benchmark}.jsonl"


def model_identity(path, revision=None):
    """Detect changed local checkpoints even when their directory name stays the same."""
    root = Path(path)
    if not root.is_dir():
        return {"hub": path, "revision": revision}
    digest = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if f.is_file() and ".git" not in f.relative_to(root).parts:
            digest.update(str(f.relative_to(root)).encode() + b"\0")
            with f.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return {"path": str(root.resolve()), "sha256": digest.hexdigest()}


def cached_cell(cell, out_path, experiment, problems, offset, n_rollouts):
    """Reject mixed or legacy experiments; resume only an exact, complete shard."""
    current = None
    for f in cell.glob("rollouts_p*.jsonl"):
        if ".tmp" in f.name:
            continue
        records = read_jsonl(f)
        if any(record.get("experiment") != experiment for record in records):
            raise SystemExit(f"Incompatible or unverified cached results in {f}. Use a new --out or --label.")
        if f == out_path:
            current = records
    if current is None:
        return False
    expected = {(p["problem_key"], i) for p in problems for i in range(offset, offset + n_rollouts)}
    actual = [(r["problem_key"], r["rollout_index"]) for r in current]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected):
        raise SystemExit(f"Cached shard layout differs in {out_path}. Use a new --out or --label.")
    return set(actual) == expected


def native_context(model_dir_or_hub, revision):
    """max_position_embeddings of the checkpoint: early checkpoints have 8,192, released models 32k+."""
    if Path(model_dir_or_hub).is_dir():
        cfg = Path(model_dir_or_hub) / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        cfg = hf_hub_download(model_dir_or_hub, "config.json", revision=revision)
    with open(cfg, encoding="utf-8") as fh:
        return int(json.load(fh).get("max_position_embeddings", 8192))


def gpu_groups(n_gpus, tp):
    """The visible GPUs in groups of tp: [["0"], ["1"], ...] or [["0", "1"], ...]."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        ids = [g.strip() for g in vis.split(",") if g.strip()]
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
            ids = [str(i) for i, line in enumerate(l for l in out.splitlines() if l.startswith("GPU"))]
        except Exception:
            ids = []
    ids = (ids or ["0"])[:n_gpus or None]
    return [ids[i:i + tp] for i in range(0, len(ids) - tp + 1, tp)] or [ids]


def _strip_args(argv, names):
    """argv without the given options and their values (both `--x v` and `--x=v`)."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in names:
            skip = True
        elif not any(a.startswith(n + "=") for n in names):
            out.append(a)
    return out


def run_split(args, n_problems, label):
    """--gpus N: one `cues eval` per GPU group on a contiguous slice of the problems, same output cells."""
    groups = gpu_groups(args.gpus, args.tensor_parallel_size)
    k = min(len(groups), n_problems)
    if len(groups) < args.gpus // args.tensor_parallel_size:
        print(f"NOTE only {len(groups) * args.tensor_parallel_size} GPUs visible; using {k} process(es)", flush=True)
    base = _strip_args(sys.argv[1:], {"--gpus", "--problem-start", "--problem-count"})
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), os.environ.get("USER", "cues"))
    procs, start = [], args.problem_start
    for w in range(k):
        count = n_problems // k + (w < n_problems % k)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(groups[w]))
        for key, d in (("TRITON_CACHE_DIR", "tri"), ("TORCHINDUCTOR_CACHE_DIR", "ind"), ("VLLM_CACHE_ROOT", "vllm")):
            env[key] = f"{tmp}/{d}_eval{w}"
        log = logs / f"{label}_{Path(args.benchmark).stem}_p{start:05d}{args.suffix}.log"
        cmd = [sys.executable, "-m", "cues", "eval", "--problem-start", str(start), "--problem-count", str(count)] + base
        procs.append((w, start, log, subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)))
        print(f"GPU {','.join(groups[w])}: problems {start}-{start + count - 1} -> {log}", flush=True)
        start += count
    failed = [(w, log) for w, _, log, p in procs if p.wait() != 0]
    if failed:
        raise SystemExit("failed: " + ", ".join(str(log) for _, log in failed))
    print(f"done on {k} process(es); `python -m cues summarize {args.out}` scores the cells", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"registry key ({', '.join(M.MODELS)}), Hub id, or a local model directory")
    ap.add_argument("--family", choices=sorted(M.FAMILIES), help="for a Hub id or directory not in the registry")
    ap.add_argument("--revision", help="Hub commit; default is the registry's pin")
    ap.add_argument("--weights", help="load these weights (e.g. a merged checkpoint) but keep --model's prompt, family and cues")
    ap.add_argument("--adapter", type=Path, help="LoRA checkpoint served on top of the model (rank <= 64)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), help="default: the registry's prompt for the model")
    ap.add_argument("--benchmark", default="math500", help=f"one of {', '.join(BENCHMARKS)}, or a JSONL path with problem_key/problem/gt")
    ap.add_argument("--cues", nargs="*", default=["none", "auto"],
                    help="none | auto (the registry's cue for this model and prompt) | a literal opening, real newlines: $'.\\n\\nOkay'")
    ap.add_argument("--cues-file", type=Path, help="JSON {name: opening} whose openings are evaluated too (the search's nominees.json)")
    ap.add_argument("--label", help="output directory name; default: the model key, plus _<adapter dir name> for an adapter")
    ap.add_argument("--chat", action="store_true", help="apply the chat template (Instruct/Think models); the cue follows the template")
    ap.add_argument("--no-think", action="store_true", help="with --chat: render with enable_thinking=False")
    ap.add_argument("--problem-start", type=int, default=0)
    ap.add_argument("--problem-count", type=int, help="default: to the end of the file")
    ap.add_argument("--n-rollouts", type=int, default=16, help="per problem in this batch (the protocol takes two batches of 16)")
    ap.add_argument("--max-new-tokens", type=int, default=31744, help="31744 for released models; 7168 for 8,192-context checkpoints")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260819, help="base seed; rollout index r is sampled with seed + r, whatever the batch")
    ap.add_argument("--rollout-index-offset", type=int, default=0, help="16 for the second batch")
    ap.add_argument("--suffix", default="", help="_b2 for the second batch, so the two never share a file")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="2 for a 32B on 80 GB cards")
    ap.add_argument("--gpus", type=int, default=1,
                    help="split the problems over this many GPUs (one vLLM process per --tensor-parallel-size GPUs)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, help="vLLM concurrency cap; default is vLLM's")
    ap.add_argument("--out", type=Path, default=RUNS / "eval", help="root of the rollout tree (default $RUNS/eval)")
    args = ap.parse_args()

    entry = M.resolve(args.model, args.family)
    prompt = args.prompt or entry["prompt"]
    chat = args.chat or bool(entry.get("chat"))
    cues = [(spec, M.parse_cue(spec, entry, prompt)) for spec in args.cues]
    if args.cues_file:
        cues += [(name, cue) for name, cue in json.load(open(args.cues_file, encoding="utf-8")).items()]
    seen = set()
    cues = [(spec, cue) for spec, cue in cues if not (cue in seen or seen.add(cue))]      # one cell per distinct opening
    if not cues:
        raise SystemExit("no cues: pass --cues and/or --cues-file")
    weights = args.weights or entry["hub"]
    revision = args.revision or (None if args.weights else entry["revision"])
    label = args.label or (entry["key"] + (f"_{args.adapter.name}" if args.adapter else ""))
    style = GRADING[prompt]

    problems = read_jsonl(problems_path(args.benchmark))
    problems = problems[args.problem_start:] if args.problem_count is None else \
        problems[args.problem_start:args.problem_start + args.problem_count]
    if not problems:
        raise SystemExit("no problems in that range")
    benchmark = problems_path(args.benchmark).stem
    if args.gpus > 1:
        return run_split(args, len(problems), label)

    if not Path(weights).is_dir() and revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(weights).sha
    native_ctx = native_context(weights, revision)
    max_model_len = min(args.max_new_tokens + 1024, native_ctx)
    max_tokens = max_model_len - 1024
    print(json.dumps({"model": weights, "revision": revision, "prompt": prompt, "benchmark": benchmark, "label": label,
                      "cues": {spec: cue for spec, cue in cues}, "native_ctx": native_ctx, "max_tokens": max_tokens,
                      "n_rollouts": args.n_rollouts, "problems": len(problems), "adapter": str(args.adapter or "")}), flush=True)
    if max_tokens != args.max_new_tokens:
        print(f"NOTE budget is {max_tokens}, not {args.max_new_tokens}: the checkpoint context is {native_ctx}", flush=True)

    # Avoid requiring FlashInfer or nvcc on the evaluation node.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    llm = LLM(model=weights, revision=revision, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max_model_len, enforce_eager=True, seed=args.seed, trust_remote_code=True,
              tensor_parallel_size=args.tensor_parallel_size,
              **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
              enable_lora=bool(args.adapter), max_lora_rank=64)
    lora = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest(args.adapter.name, 1, str(args.adapter))
    tok = llm.get_tokenizer()
    # the family's end-of-turn tokens (Qwen3 base closes with <|im_end|>, which its tokenizer does not call EOS)
    stop_ids = [tok.convert_tokens_to_ids(t) for t in M.FAMILIES[entry["family"]]["stop_tokens"]]
    stop_ids = [i for i in stop_ids if isinstance(i, int) and i >= 0 and i != tok.unk_token_id]

    experiment_base = {
        "schema": 1, "model": model_identity(weights, revision),
        "adapter": model_identity(str(args.adapter)) if args.adapter else None,
        "prompt": PROMPTS[prompt], "chat": chat, "no_think": args.no_think,
        "family": entry["family"], "stop_ids": stop_ids, "stops": STOPS[prompt],
        "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
        "max_tokens": max_tokens, "grading": style,
        "dataset_sha256": hashlib.sha256(problems_path(args.benchmark).read_bytes()).hexdigest(),
    }
    for spec, cue in cues:
        experiment = dict(experiment_base, cue=cue)
        cell = args.out / label / prompt / benchmark / M.cue_slug(cue)
        cell.mkdir(parents=True, exist_ok=True)
        out_path = cell / f"rollouts_p{args.problem_start:05d}{args.suffix}.jsonl"
        if cached_cell(cell, out_path, experiment, problems, args.rollout_index_offset, args.n_rollouts):
            print(json.dumps({"cue": spec, "skipped": str(out_path)}), flush=True)
            continue
        if chat:
            extra = {"enable_thinking": False} if args.no_think else {}
            prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPTS[prompt].format(q=p["problem"])}],
                                               tokenize=False, add_generation_prompt=True, **extra) + cue for p in problems]
        else:
            prompts = [render(prompt, p["problem"], cue) for p in problems]
        params = SamplingParams(n=args.n_rollouts, temperature=args.temperature, top_p=args.top_p, max_tokens=max_tokens,
                                stop=STOPS[prompt] or None, stop_token_ids=stop_ids or None,
                                seed=args.seed + args.rollout_index_offset)
        outputs = llm.generate(prompts, params, lora_request=lora)

        for stale in cell.glob(out_path.name + ".tmp*"):
            stale.unlink(missing_ok=True)
        tmp = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
        n_correct, n = 0, len(problems) * args.n_rollouts
        with tmp.open("w", encoding="utf-8") as sink:
            for problem, out in zip(problems, outputs):
                for index, cand in enumerate(out.outputs):
                    completion = cue + cand.text
                    verdict = is_correct(problem["gt"], completion, style)
                    n_correct += verdict is True
                    sink.write(json.dumps({
                        "problem_key": problem["problem_key"], "rollout_index": index + args.rollout_index_offset,
                        "cue": cue, "label": label, "prompt": prompt, "experiment": experiment,
                        "committed": verdict is not None, "correct": verdict is True,
                        "hit_token_cap": cand.finish_reason == "length", "finish_reason": cand.finish_reason,
                        "output_tokens": len(cand.token_ids), "max_tokens_used": max_tokens,
                        "completion": completion}, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        print(json.dumps({"cue": spec, "wrote": str(out_path), "pass@1": round(n_correct / n, 4)}), flush=True)

    # Outputs are saved. Bound engine shutdown to avoid hanging the parent job.
    import threading
    watchdog = threading.Timer(300, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    llm.llm_engine.engine_core.shutdown(timeout=120)


if __name__ == "__main__":
    main()
