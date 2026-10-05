"""Propose openings from the model's own boundary distribution: a depth-2 beam over next tokens.

    python -m cues beam --model qwen14b --prompt boxed                # writes $RUNS/search/qwen14b_boxed/beam.json

Step 1 of the label-free discovery (paper §2.1). At the boundary where the solution starts, take the
top-k next tokens, extend each by its top continuations, and keep the `beam-width` most probable 1- and
2-token openings per problem. Pool them across probe problems (100 MATH-train problems, disjoint from
every benchmark) and keep those present on at least half of them, ranked by mean probability mass.
This needs only the model: no rollouts, no labels, no training data. The top 20 go on to the screen
(cues.search); nothing here decides.

Output JSON carries the settings and `candidates`: [{prefix, kind (gram1|gram2), problem_fraction,
mean_prob_mass}] in mass order. Also works on a merged RL checkpoint (--weights) to read what the
policy learned to open with.

Provenance: sophicle/reason registers/beam.py.
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cues import models as M
from cues.evaluate import RUNS, problems_path, read_jsonl
from cues.prompts import PROMPTS

TEMPLATES = dict(PROMPTS)
TEMPLATES.update({"bare": "{q}\n", "qa": "Q: {q}\nA:"})       # two format-free boundaries, for probing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry key, Hub id or directory")
    ap.add_argument("--family", choices=sorted(M.FAMILIES))
    ap.add_argument("--revision")
    ap.add_argument("--weights", help="load these weights (a merged checkpoint) under --model's registry entry")
    ap.add_argument("--prompt", choices=sorted(TEMPLATES), help="the boundary to probe; default: the registry's prompt")
    ap.add_argument("--chat", action="store_true", help="probe after the chat template's generation prompt")
    ap.add_argument("--benchmark", default="math_train_probe", help="problems file; the probe set by default")
    ap.add_argument("--problem-start", type=int, default=0)
    ap.add_argument("--problems", type=int, default=100)
    ap.add_argument("--top-first", type=int, default=12, help="first-token candidates per problem")
    ap.add_argument("--top-second", type=int, default=4, help="continuations expanded per surviving prefix")
    ap.add_argument("--depth", type=int, default=2, help="how many tokens an opening may grow to")
    ap.add_argument("--beam-width", type=int, default=40, help="partial openings kept between depths")
    ap.add_argument("--min-problem-fraction", type=float, default=0.5,
                    help="keep openings in the beam on at least this fraction of problems")
    ap.add_argument("--per-problem", action="store_true", help="also store each opening's mass on every problem")
    ap.add_argument("--out", type=Path, help="default $RUNS/search/<model>_<prompt>/beam.json")
    args = ap.parse_args()

    entry = M.resolve(args.model, args.family)
    prompt = args.prompt or entry["prompt"]
    weights = args.weights or entry["hub"]
    revision = args.revision or (None if args.weights else entry["revision"])
    out = args.out or RUNS / "search" / f"{entry['key']}_{prompt}" / "beam.json"

    tokenizer = AutoTokenizer.from_pretrained(weights, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(weights, revision=revision, dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True).eval()
    problems = read_jsonl(problems_path(args.benchmark))[args.problem_start:args.problem_start + args.problems]

    presence = collections.defaultdict(collections.Counter)      # depth -> opening -> problems it was in the beam on
    mass = collections.defaultdict(collections.Counter)          # depth -> opening -> summed beam mass
    per_problem = []
    for problem in problems:
        pmass = collections.Counter()
        per_problem.append(pmass)
        text = TEMPLATES[prompt].format(q=problem["problem"])
        if args.chat:
            text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=not args.chat)["input_ids"].to(model.device)
        beams = [(ids, 1.0, "")]
        for step in range(args.depth):
            scored = []
            for ids, p_prefix, opening in beams:
                with torch.no_grad():
                    probs = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)
                top = torch.topk(probs, args.top_first if step == 0 else args.top_second)
                for p, t in zip(top.values.tolist(), top.indices.tolist()):
                    nxt = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
                    scored.append((nxt, p_prefix * p, opening + tokenizer.decode([t])))
            scored.sort(key=lambda x: -x[1])
            beams = scored[:args.beam_width]
            for _, p, opening in beams:
                presence[step + 1][opening] += 1
                mass[step + 1][opening] += p
                pmass[(step + 1, opening)] += p

    n = len(problems)
    candidates = []
    for depth, seen in presence.items():
        for opening, cnt in seen.items():
            if cnt >= args.min_problem_fraction * n:
                c = {"prefix": opening, "kind": f"gram{depth}", "problem_fraction": round(cnt / n, 3),
                     "mean_prob_mass": round(mass[depth][opening] / n, 5)}
                if args.per_problem:
                    c["per_problem_mass"] = [round(pm[(depth, opening)], 7) for pm in per_problem]
                candidates.append(c)
    candidates.sort(key=lambda c: -c["mean_prob_mass"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": entry["key"], "weights": weights, "revision": revision, "prompt": prompt,
                               "chat": args.chat, "benchmark": args.benchmark, "problem_start": args.problem_start,
                               "problems": n, "depth": args.depth, "beam_width": args.beam_width,
                               "top_first": args.top_first, "top_second": args.top_second,
                               "min_problem_fraction": args.min_problem_fraction,
                               "candidates": candidates}, indent=1) + "\n", encoding="utf-8")
    for c in candidates[:25]:
        print(f"  {c['kind']:6s} {c['prefix']!r:28s} mass={c['mean_prob_mass']:.4f} on {c['problem_fraction']:.0%} of problems")
    print(f"{len(candidates)} openings -> {out}")


if __name__ == "__main__":
    main()
