"""Find a model's cue end to end: beam -> nominees -> screen -> decider (paper §2.1, App. A).

    python -m cues search --model qwen14b --prompt boxed              # ~2-3 GPU-hours on a 14B
    python -m cues search --model olmo7b                              # the registry prompt (rlzero)
    python -m cues search --model my-org/my-base --family qwen --prompt boxed

Everything lands under --out (default $RUNS/search/<model>_<prompt>):
  beam.json       cues.beam on 100 probe problems (data/problems/math_train_probe.jsonl[0:100])
  nominees.json   the 20 openings with the largest mean beam mass, {slug: opening}
  screen/         cues.evaluate on probe problems 0-29: no cue + every nominee, 16 rollouts x 16,384 tokens
  decider.json    cues.agree over the screen (entropy with NONE as a symbol, the silence guard)
  winner.json     the selected cue, its entropy against the no-cue entropy, and the accuracy check
The screen is the cost (21 openings x 480 rollouts at a 16k budget): it is spread over the visible GPUs,
one vLLM process per GPU (or per --tensor-parallel-size group), about 4 GPU-hours for Qwen3-14B and
~25 for Olmo-3-7B, whose base rollouts run long. A step whose output exists is skipped, so the search
can be resumed; delete a file to redo it. The probe problems are MATH-train problems disjoint from every
benchmark; their answers are used only for the accuracy column, never for the choice.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cues import models as M
from cues.agree import decide, report, score_screen
from cues.evaluate import RUNS, gpu_groups
from cues.prompts import GRADING, PROMPTS

REPO = Path(__file__).resolve().parent.parent


def env_for(gpus=None, worker=None):
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if worker is not None:       # node-local compile caches, one set per process
        tmp = os.path.join(tempfile.gettempdir(), os.environ.get("USER", "cues"))
        for k, d in (("TRITON_CACHE_DIR", "tri"), ("TORCHINDUCTOR_CACHE_DIR", "ind"), ("VLLM_CACHE_ROOT", "vllm")):
            env[k] = f"{tmp}/{d}_search{worker}"
            os.makedirs(env[k], exist_ok=True)
    return env


def run(cmd):
    print("+", " ".join(repr(c) if any(ch in c for ch in " \n") else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env_for())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=sorted(M.FAMILIES))
    ap.add_argument("--prompt", choices=sorted(PROMPTS), help="default: the registry's prompt")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-nominees", type=int, default=20)
    ap.add_argument("--beam-problems", type=int, default=100)
    ap.add_argument("--screen-problems", type=int, default=30)
    ap.add_argument("--screen-rollouts", type=int, default=16)
    ap.add_argument("--screen-budget", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--guard", type=float, default=0.10)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpus", type=int, help="GPUs to spread the screen over (default: all visible), in groups of --tensor-parallel-size")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--beam-only", action="store_true", help="stop after nominees.json")
    a = ap.parse_args()

    entry = M.resolve(a.model, a.family)
    prompt = a.prompt or entry["prompt"]
    out = a.out or RUNS / "search" / f"{entry['key']}_{prompt}"
    out.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-m", "cues"]
    ident = ["--model", a.model] + (["--family", a.family] if a.family else [])

    beam = out / "beam.json"
    if not beam.exists():
        run(py + ["beam"] + ident + ["--prompt", prompt, "--problems", str(a.beam_problems), "--out", str(beam)])
    cands = json.load(open(beam))["candidates"]

    nominees_path = out / "nominees.json"
    if not nominees_path.exists():
        top = sorted(cands, key=lambda c: -c["mean_prob_mass"])[:a.n_nominees]
        nominees = {"none": ""}
        for c in top:
            nominees.setdefault(M.cue_slug(c["prefix"]), c["prefix"])
        nominees_path.write_text(json.dumps(nominees, indent=1) + "\n")
    nominees = json.load(open(nominees_path))
    print(f"{len(nominees) - 1} nominees:", ", ".join(repr(v) for k, v in nominees.items() if k != "none"), flush=True)
    if a.beam_only:
        return

    # Each GPU group evaluates every k-th opening.
    screen_root = out / "screen"
    screen_root.mkdir(parents=True, exist_ok=True)
    groups = gpu_groups(a.gpus, a.tensor_parallel_size)
    items = list(nominees.items())                       # no cue first, then the nominees in beam order
    procs = []
    for w, group in enumerate(groups):
        mine = dict(items[w::len(groups)])
        if not mine:
            continue
        sub = screen_root / f"nominees_w{w}.json"
        sub.write_text(json.dumps(mine, indent=1) + "\n")
        log = screen_root / f"worker{w}.log"
        cmd = py + ["eval"] + ident + ["--prompt", prompt, "--benchmark", "math_train_probe", "--label", "screen",
                    "--problem-start", "0", "--problem-count", str(a.screen_problems),
                    "--n-rollouts", str(a.screen_rollouts), "--max-new-tokens", str(a.screen_budget), "--seed", str(a.seed),
                    "--tensor-parallel-size", str(a.tensor_parallel_size), "--gpu-memory-utilization", str(a.gpu_memory_utilization),
                    "--out", str(screen_root), "--cues", "--cues-file", str(sub)]
        print(f"+ screen worker {w} on GPU {','.join(group)}: {len(mine)} openings, log {log}", flush=True)
        procs.append((w, subprocess.Popen(cmd, env=env_for(group, w), stdout=open(log, "w"), stderr=subprocess.STDOUT)))
    failed = [w for w, proc in procs if proc.wait() != 0]
    if failed:
        raise SystemExit(f"screen worker(s) {failed} failed; see {screen_root}/worker*.log")

    screen_dir = screen_root / "screen" / prompt / "math_train_probe"
    scores = score_screen(screen_dir, GRADING[prompt], a.screen_rollouts)
    winner = decide(scores, a.guard)
    report(scores, winner)
    (out / "decider.json").write_text(json.dumps({"screen": str(screen_dir), "prompt": prompt, "guard": a.guard,
                                                  "winner": winner, "scores": scores}, indent=1) + "\n")
    if winner is None:
        print("no nominee passed the silence guard; the model has no cue at this boundary by this method")
        return
    mass = {M.cue_slug(c["prefix"]): c["mean_prob_mass"] for c in cands}
    w = scores[winner]
    base = scores.get("none", {})
    result = {"model": entry["key"], "hub": entry["hub"], "prompt": prompt, "cue": w["cue"], "cue_slug": winner,
              "entropy": w["entropy"], "no_cue_entropy": base.get("entropy"),
              "accuracy": w["accuracy"], "no_cue_accuracy": base.get("accuracy"),
              "none_share": w["none_share"], "no_cue_none_share": base.get("none_share"), "guard": a.guard,
              "nominees": [{"slug": k, "cue": s["cue"], "beam_mass": mass.get(k), "entropy": s["entropy"],
                            "none_share": s["none_share"], "accuracy": s["accuracy"], "eligible": s["eligible"]}
                           for k, s in sorted(scores.items(), key=lambda kv: kv[1]["entropy"]) if k != "none"]}
    (out / "winner.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwinner for {entry['key']} under {prompt}: {w['cue']!r}  entropy {w['entropy']:.3f} vs {base.get('entropy')} with no cue"
          f"  (accuracy {w['accuracy']:.3f} vs {base.get('accuracy')})\n-> {out / 'winner.json'}\n"
          f"evaluate it: python -m cues eval --model {a.model} --prompt {prompt} --cues none {json.dumps(w['cue'])}")


if __name__ == "__main__":
    main()
