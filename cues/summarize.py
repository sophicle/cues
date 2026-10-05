"""Score every cell under a rollout tree, and print them as a table.

    python -m cues summarize $RUNS/eval                       # writes <cell>/summary.json for every cell
    python -m cues table $RUNS/eval --benchmark math500       # rows = model/prompt, columns = cue
    python -m cues table $RUNS/eval --benchmark math500 --long

A cell is <root>/<label>/<prompt>/<benchmark>/<cue slug>/ holding rollouts_p*.jsonl shards from
cues.evaluate. Scores come from cues.metrics (pass@1 with its bootstrap interval over problems, pass@8,
pass@16, the SD over rollout index), plus the cap rate, the committed rate and token counts. HumanEval
cells are scored from their <cue slug>_execgraded sibling (cues.exec_grade), never from the stored
string verdict. `complete` says whether every problem of the benchmark has --rollouts rollouts at one
budget; an incomplete cell is still scored and marked.

Provenance: sophicle/reason scripts/eval/summarize_cell.py and scripts/eval/collab_table.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from cues.evaluate import DATA, RUNS, read_jsonl
from cues.metrics import scores_from_records


def expected_problems(benchmark):
    p = DATA / f"{benchmark}.jsonl"
    return sum(1 for line in open(p, encoding="utf-8") if line.strip()) if p.is_file() else None


def load_cell(cell):
    """Records of a cell, one per (problem_key, rollout_index); the first file to name a rollout wins."""
    seen, records = set(), []
    for f in sorted(cell.glob("rollouts_p*.jsonl")):
        if ".tmp" in f.name:
            continue
        for r in read_jsonl(f):
            k = (r["problem_key"], r["rollout_index"])
            if k not in seen:
                seen.add(k)
                records.append(r)
    return records


def summarize_cell(cell, rollouts):
    label, prompt, benchmark, slug = cell.parts[-4:]
    source = cell
    if benchmark == "humaneval":
        source = cell.with_name(cell.name + "_execgraded")
        if not source.is_dir():
            return None
    records = load_cell(source)
    if not records:
        return None
    if benchmark == "humaneval":
        for r in records:
            r["correct"] = bool(r.get("exec_correct"))
    s = scores_from_records(records)
    per = {}
    for r in records:
        per[r["problem_key"]] = per.get(r["problem_key"], 0) + 1
    budgets = sorted({r.get("max_tokens_used") for r in records})
    n_exp = expected_problems(benchmark)
    s.update({"label": label, "prompt": prompt, "benchmark": benchmark, "cue_slug": slug,
              "cue": records[0].get("cue", ""), "expected_problems": n_exp,
              "rollouts_per_problem": [min(per.values()), max(per.values())], "budget": budgets,
              "complete": (n_exp is not None and len(per) == n_exp and min(per.values()) >= rollouts and len(budgets) == 1),
              "graded_by": "execution" if benchmark == "humaneval" else "answer",
              "files": sorted(f.name for f in source.glob("rollouts_p*.jsonl") if ".tmp" not in f.name)})
    return s


def cells_under(root, label=None):
    for f in sorted(Path(root).glob(f"{label or '*'}/*/*/*/rollouts_p*.jsonl")):
        if ".tmp" in f.name or f.parent.name.endswith("_execgraded"):
            continue
        yield f.parent


def summarize(root, rollouts, label=None):
    done, n = set(), 0
    for cell in cells_under(root, label):
        if cell in done:
            continue
        done.add(cell)
        s = summarize_cell(cell, rollouts)
        if s is None:
            print(f"skip {cell}: nothing to score" + (" (run `cues exec-grade` first)" if cell.parts[-2] == "humaneval" else ""),
                  file=sys.stderr)
            continue
        (cell / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        n += 1
        print(f"{s['label']}/{s['prompt']}/{s['benchmark']}/{s['cue_slug']}: pass@1 {s['pass@1']:.4f} "
              f"[{s['pass@1_ci95'][0]:.3f}, {s['pass@1_ci95'][1]:.3f}]  cap {s['cap_rate']:.3f}  "
              f"{s['n_problems']}/{s['expected_problems']} problems x {s['rollouts_per_problem']} rollouts"
              + ("" if s["complete"] else "  (incomplete)"))
    print(f"{n} cells summarized under {root}")


def load_summaries(root, benchmark):
    out = []
    for f in sorted(Path(root).glob(f"*/*/{benchmark}/*/summary.json")):
        out.append(json.load(open(f, encoding="utf-8")))
    return out


def fmt_cue(slug, cue):
    return "no cue" if slug == "none" else repr(cue)


def table(root, benchmark, long=False):
    rows = load_summaries(root, benchmark)
    if not rows:
        raise SystemExit(f"no summaries for {benchmark} under {root}; run `python -m cues summarize {root}` first")
    if long:
        print("| model | prompt | cue | pass@1 | 95% CI | SD rollouts | pass@8 | pass@16 | cap rate | committed | tokens | rollouts | complete |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        f3 = lambda x: "-" if x is None else f"{x:.3f}"
        for s in sorted(rows, key=lambda s: (s["label"], s["prompt"], s["cue_slug"] != "none", s["cue_slug"])):
            print(f"| {s['label']} | {s['prompt']} | `{fmt_cue(s['cue_slug'], s['cue'])}` | {s['pass@1']:.3f} | "
                  f"{s['pass@1_ci95'][0]:.3f}-{s['pass@1_ci95'][1]:.3f} | {f3(s['pass@1_sd_rollouts'])} | {f3(s['pass@8'])} | "
                  f"{f3(s['pass@16'])} | {s['cap_rate']:.3f} | {s['committed']:.3f} | {s['mean_output_tokens']:.0f} | "
                  f"{s['rollouts_per_problem'][0]} | {'yes' if s['complete'] else 'no'} |")
        return
    slugs = sorted({s["cue_slug"] for s in rows}, key=lambda x: (x != "none", x))
    names = {s["cue_slug"]: fmt_cue(s["cue_slug"], s["cue"]) for s in rows}
    grid = {}
    for s in rows:
        grid.setdefault((s["label"], s["prompt"]), {})[s["cue_slug"]] = s
    print(f"{benchmark}: pass@1 [95% CI over problems]; * = incomplete cell\n")
    print("| model | prompt | " + " | ".join(f"`{names[c]}`" for c in slugs) + " |")
    print("|---|---|" + "---|" * len(slugs))
    for (label, prompt), by in sorted(grid.items()):
        cells = []
        for c in slugs:
            s = by.get(c)
            cells.append("" if s is None else f"{s['pass@1']:.3f} [{s['pass@1_ci95'][0]:.2f}, {s['pass@1_ci95'][1]:.2f}]"
                         + ("" if s["complete"] else "*"))
        print(f"| {label} | {prompt} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=str(RUNS / "eval"))
    ap.add_argument("--rollouts", type=int, default=32, help="rollouts per problem a complete cell needs")
    ap.add_argument("--label", help="summarize only this model label")
    ap.add_argument("--table", action="store_true", help="print the table instead of writing summaries")
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--long", action="store_true", help="one row per cell with every score")
    a = ap.parse_args()
    if a.table:
        table(a.root, a.benchmark, a.long)
    else:
        summarize(a.root, a.rollouts, a.label)


if __name__ == "__main__":
    main()
