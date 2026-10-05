"""python -m cues <command> [args...]

Each command is one module with its own --help.
"""
import runpy
import sys

COMMANDS = {
    "demo":       ("cues.demo",             [],          "try no cue vs auto on five MATH-500 problems; not a benchmark run"),
    "eval":       ("cues.evaluate",         [],          "sample rollouts for a model x prompt x benchmark x cues (vLLM)"),
    "summarize":  ("cues.summarize",        [],          "write summary.json (pass@k, CIs, cap rate) for every cell under a runs root"),
    "table":      ("cues.summarize",        ["--table"], "print a markdown table of the summaries for one benchmark"),
    "exec-grade": ("cues.exec_grade",       [],          "grade HumanEval rollouts by running the tests"),
    "beam":       ("cues.beam",             [],          "propose openings from the model's boundary distribution (depth-2 beam)"),
    "agree":      ("cues.agree",            [],          "rank openings by answer entropy over their screen rollouts (label-free)"),
    "search":     ("cues.search",           [],          "beam -> screen -> decider: find a model's cue end to end"),
    "train":      ("cues.rl.train",         [],          "GRPO on MATH train with the truncation penalty, cue optional"),
    "merge":      ("cues.rl.merge",         [],          "merge a LoRA checkpoint into full weights (needed for the 32B evals)"),
    "openers":    ("cues.rl.opener_share",  [],          "opening-word shares and paired premium over a training run"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        width = max(len(c) for c in COMMANDS)
        print(__doc__)
        for cmd, (_, _, desc) in COMMANDS.items():
            print(f"  {cmd:<{width}}  {desc}")
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        raise SystemExit(f"unknown command {cmd!r}; run `python -m cues --help`")
    module, extra, _ = COMMANDS[cmd]
    sys.argv = [f"cues {cmd}"] + extra + sys.argv[2:]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
