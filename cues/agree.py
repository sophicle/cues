"""Rank openings label-free by the entropy of the answers they lead to: the decider (paper §2.1, App. A.1).

    python -m cues agree $RUNS/search/qwen14b_boxed/screen/screen/boxed/math_train_probe

The argument is a <prompt>/<benchmark> directory of cues.evaluate cells, one per opening (the screen:
no cue plus the beam's nominees, 16 rollouts per problem). For each opening, per problem, the answer
of every rollout is extracted with the grader and reduced to an equivalence key (0.5 and 1/2 agree);
"no answer" is its own symbol, NONE. Per opening:

  entropy     mean over problems of the Shannon entropy of that answer distribution; the decider
  none_share  share of rollouts with no answer; an opening whose none_share exceeds the no-cue
              cell's by more than --guard is dropped, so consistent silence cannot pose as convergence
  agreement   share of rollouts on the modal answer, for comparison
  accuracy    the check the method never sees (the probe set carries answers)

The winner is the eligible opening with the lowest entropy. Writes decider.json when --out is given.

Provenance: sophicle/reason registers/agree.py.
"""
import argparse
import collections
import json
import math
from pathlib import Path

from cues.evaluate import read_jsonl
from cues.grading import _parse, extract
from cues.prompts import GRADING


def answer_key(text, style):
    """A hashable key for 'the same answer': a rounded float when the answer is numeric, else the grader's
    normalised string (the second element of _parse), as in the paper's decider. Only the tail of a completion
    holds the answer, and parsing can stall on pathological strings, so the text is cut to its last 2,000
    characters and the answer to 80."""
    a = extract(text[-2000:], style)
    if a is None:
        return None
    try:
        v = _parse(a[:80])
    except Exception:
        return None
    if not v:
        return None
    num, txt = (list(v) + [None, None])[:2]
    try:
        return round(float(num), 6)
    except Exception:
        return str(txt if txt is not None else num)


def score_cell(cell, style, min_rollouts=16):
    answers = collections.defaultdict(dict)
    correct = collections.defaultdict(dict)
    cue = ""
    for f in sorted(Path(cell).glob("rollouts_p*.jsonl")):
        if ".tmp" in f.name:
            continue
        for r in read_jsonl(f):
            answers[r["problem_key"]][r["rollout_index"]] = answer_key(r.get("completion", ""), style)
            correct[r["problem_key"]][r["rollout_index"]] = bool(r["correct"])
            cue = r.get("cue", "")
    ents, nones, agrees, accs = [], [], [], []
    for pk, rolls in answers.items():
        if len(rolls) < min_rollouts:
            continue
        cnt = collections.Counter("NONE" if v is None else v for v in rolls.values())
        tot = sum(cnt.values())
        ents.append(-sum(c / tot * math.log2(c / tot) for c in cnt.values()))
        nones.append(cnt.get("NONE", 0) / tot)
        vals = [v for v in rolls.values() if v is not None]
        agrees.append(collections.Counter(vals).most_common(1)[0][1] / tot if vals else 0.0)
        accs.append(sum(correct[pk].values()) / len(correct[pk]))
    n = len(ents)
    if not n:
        return None
    mean_ent = sum(ents) / n
    sd = (sum((e - mean_ent) ** 2 for e in ents) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"cue": cue, "n": n, "entropy": round(mean_ent, 4), "entropy_sd": round(sd, 4),
            "entropies": [round(e, 3) for e in ents], "none_share": round(sum(nones) / n, 4),
            "agreement": round(sum(agrees) / n, 4), "accuracy": round(sum(accs) / n, 4)}


def score_screen(screen_dir, style, min_rollouts=16):
    """{cue slug: scores} for every cell directory under <prompt>/<benchmark>."""
    out = {}
    for cell in sorted(Path(screen_dir).iterdir()):
        if cell.is_dir() and not cell.name.endswith("_execgraded"):
            s = score_cell(cell, style, min_rollouts)
            if s:
                out[cell.name] = s
    return out


def decide(scores, guard=0.10):
    """Mark eligibility and return the winning slug (lowest entropy among the eligible), or None."""
    base = scores.get("none")
    for slug, s in scores.items():
        s["eligible"] = slug != "none" and (base is None or s["none_share"] <= base["none_share"] + guard)
    eligible = [k for k, s in scores.items() if s["eligible"]]
    return min(eligible, key=lambda k: (scores[k]["entropy"], k)) if eligible else None


def report(scores, winner):
    print(f"  {'cue':28s} {'n':>3}  {'entropy':>14}  {'none':>5}  {'agree':>5}  {'[acc]':>6}")
    for slug, s in sorted(scores.items(), key=lambda kv: kv[1]["entropy"]):
        mark = "<- winner" if slug == winner else ("" if s["eligible"] or slug == "none" else "(guard: silent)")
        name = "no cue" if slug == "none" else repr(s["cue"])
        print(f"  {name:28s} {s['n']:3d}  {s['entropy']:6.3f} +- {s['entropy_sd']:5.3f}  {s['none_share']:5.2f}  "
              f"{s['agreement']:5.3f}  {s['accuracy']:6.3f}  {mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("screen_dir", help="a <prompt>/<benchmark> directory of cells, one per opening")
    ap.add_argument("--prompt", choices=sorted(GRADING), help="which answer contract to grade by; default: the directory's prompt")
    ap.add_argument("--min-rollouts", type=int, default=16, help="problems with fewer rollouts are ignored")
    ap.add_argument("--guard", type=float, default=0.10, help="allowed none_share excess over the no-cue cell")
    ap.add_argument("--out", type=Path, help="write decider.json here")
    a = ap.parse_args()
    prompt = a.prompt or Path(a.screen_dir).resolve().parent.name
    scores = score_screen(a.screen_dir, GRADING[prompt], a.min_rollouts)
    if not scores:
        raise SystemExit(f"no scorable cells under {a.screen_dir} (need >= {a.min_rollouts} rollouts per problem)")
    winner = decide(scores, a.guard)
    report(scores, winner)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"screen": str(a.screen_dir), "prompt": prompt, "guard": a.guard,
                                     "winner": winner, "scores": scores}, indent=1) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
