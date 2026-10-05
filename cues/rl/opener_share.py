"""How the opening word of training rollouts moves over a run, from <run>/rollout_log (paper App. D.2).

    python -m cues openers $RUNS/rl/olmo7b [--window 50] [--top 8]

Per window of steps: the share of rollouts by opening word, the mean reward given that opener, and the
paired within-problem premium (mean over problems of reward|opener minus reward|other, over problems
that produced both), which is what says an opener is being selected for rather than merely growing.

Provenance: avdravid/reasoning_registers_grpo GRPO_clean scripts/opener_share.py.
"""
import argparse
import collections
import glob
import json
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for f in glob.glob(f"{a.run}/rollout_log/rollouts_rank*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r.get("gstep") is not None:
                r["reward"] = r["reward"] or 0.0       # a dropped (None) rollout counts as 0 here
                rows.append(r)
    if not rows:
        raise SystemExit(f"no rollout records under {a.run}/rollout_log")
    byw = collections.defaultdict(list)
    for r in rows:
        byw[r["gstep"] // a.window].append(r)
    for w in sorted(byw):
        rs = byw[w]
        n = len(rs)
        print(f"\n=== steps {w * a.window + 1}-{(w + 1) * a.window}  n={n}  reward {statistics.mean(r['reward'] for r in rs):.3f}  "
              f"capped {statistics.mean(r['capped'] for r in rs):.3f}  tokens {statistics.mean(r['n_tokens'] for r in rs):.0f}")
        print(f"   {'share':>6} {'reward':>7} {'paired':>7}  opener")
        by_op = collections.defaultdict(list)
        for r in rs:
            by_op[r["opener"]].append(r)
        for op, xs in sorted(by_op.items(), key=lambda kv: -len(kv[1]))[:a.top]:
            prob = collections.defaultdict(lambda: ([], []))
            for r in rs:
                prob[r["pid"]][0 if r["opener"] == op else 1].append(r["reward"])
            pairs = [(statistics.mean(x), statistics.mean(y)) for x, y in prob.values() if x and y]
            paired = statistics.mean(x - y for x, y in pairs) if pairs else float("nan")
            print(f"   {len(xs) / n:6.3f} {statistics.mean(r['reward'] for r in xs):7.3f} {paired:+7.3f}  {op!r}")


if __name__ == "__main__":
    main()
