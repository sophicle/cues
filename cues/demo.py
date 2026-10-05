"""Compare no cue and the model's selected cue on five MATH-500 problems.

This is a setup check, not the paper's evaluation protocol. The model still
requires a supported GPU and its normal weight download.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from cues import models


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=[
        key for key, entry in models.MODELS.items() if entry.get("trainable")
    ])
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=1,
                        help="GPUs for the model; use enough memory to load its weights")
    parser.add_argument("--out", type=Path,
                        default=Path(os.environ.get("RUNS", "runs")) / "demo",
                        help="output root, separate from full evaluations (default: $RUNS/demo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the plan without downloading a model or running evaluation")
    return parser


def commands(args):
    prefix = [sys.executable, "-m", "cues"]
    return [
        prefix + ["eval", "--model", args.model, "--benchmark", "math500",
                  "--cues", "none", "auto", "--problem-count", "5",
                  "--n-rollouts", "1", "--max-new-tokens", "2048",
                  "--tensor-parallel-size", str(args.tensor_parallel_size),
                  "--out", str(args.out)],
        prefix + ["summarize", str(args.out), "--rollouts", "1", "--label", args.model],
        prefix + ["table", str(args.out), "--benchmark", "math500"],
    ]


def main(argv=None):
    args = build_parser().parse_args(argv)
    entry = models.resolve(args.model)
    cue = models.cue_for(entry, entry["prompt"])
    plan = commands(args)
    print(f"Demo: {args.model} ({entry['hub']})\n"
          f"Prompt: {entry['prompt']} | Cues: no cue vs {cue!r}\n"
          f"5 MATH-500 problems × 1 rollout × 2 cues; up to 2,048 new tokens each\n"
          f"GPUs: {args.tensor_parallel_size} | Output: {args.out}\n"
          "This checks the workflow, not benchmark accuracy. Model memory requirements still apply.",
          flush=True)
    if args.dry_run:
        for command in plan:
            print(shlex.join(command))
        return
    for command in plan:
        result = subprocess.run(command)
        if result.returncode:
            print(f"Stopped: {shlex.join(command)}\n"
                  "Existing outputs are preserved; retry the same command to resume.", file=sys.stderr)
            raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
    print(f"\nSaved rollouts and summaries under {args.out / args.model}.\n"
          "The table marks these five-problem cells incomplete because they are not a full benchmark.\n"
          f"View results again: {shlex.join(plan[-1])}", flush=True)


if __name__ == "__main__":
    main()
