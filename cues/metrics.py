"""The one definition of every score in the paper: pass@k, the SD over rollout index, and the bootstrap
over problems. Everything that reports a number (summarize_cell.py, collab_table.py, build_grid.py,
build_eval_token_cues.py, scripts/engaging/summarize.py, registers/coverage.py) imports from here; nothing
keeps its own copy (Sophie, 2026-09-05).

Input everywhere: correct_by_problem = {problem_key: [0/1, ...] in rollout-index order}.

- pass@1: mean over problems of the per-problem accuracy (mean over its rollouts).
- pass@k: unbiased estimator from n samples with c correct, 1 - C(n-c, k)/C(n, k) (Chen et al. 2021),
  averaged over problems; None if any problem has fewer than k rollouts.
- pass@1_sd_rollouts: sample SD (n-1) of the per-rollout-index accuracies, index i = accuracy of rollout i
  over all problems (the "SD across the 32 traces").
- *_sd_boot and *_ci95: over BOOT = 1,000 resamples of the problem set, random.Random(SEED = 0), one rng
  per cell, drawn in the order boot(pass@1), boot(pass@8), boot(pass@16). ci95 = the 2.5th and 97.5th
  percentile of the resampled means; sd_boot = their sample SD.
"""
import math
import random

BOOT = 1000
SEED = 0
KS = (8, 16)


def pass_at_k(n: int, c: int, k: int):
    """Unbiased pass@k for one problem with n rollouts and c correct; None when n < k."""
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    """Sample SD (n-1); 0 for a single value."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def bootstrap(per_problem, rng, boot=BOOT):
    """(ci95, sd) of the mean over `boot` resamples of the problems, with replacement, using `rng`."""
    n = len(per_problem)
    means = sorted(mean(rng.choices(per_problem, k=n)) for _ in range(boot))
    return [round(means[int(0.025 * boot)], 4), round(means[int(0.975 * boot) - 1], 4)], round(sd(means), 4)


def per_problem_pass1(correct_by_problem):
    return [mean(v) for v in correct_by_problem.values()]


def per_problem_pass_at_k(correct_by_problem, k):
    vals = []
    for v in correct_by_problem.values():
        p = pass_at_k(len(v), sum(v), k)
        if p is None:
            return None
        vals.append(p)
    return vals


def by_rollout_index(correct_by_problem):
    """Accuracy of rollout i over the problems that have it, for every index present, in index order."""
    idx = sorted({i for v in correct_by_problem.values() for i in range(len(v))})
    return [round(mean([v[i] for v in correct_by_problem.values() if i < len(v)]), 4) for i in idx]


def cell_scores(correct_by_problem, ks=KS, seed=SEED):
    """Every score for one cell, keys as written in the cell summaries."""
    if not correct_by_problem:
        return {}
    rng = random.Random(seed)
    per1 = per_problem_pass1(correct_by_problem)
    ci, s = bootstrap(per1, rng)
    idx = by_rollout_index(correct_by_problem)
    out = {"pass@1": round(mean(per1), 4), "pass@1_ci95": ci, "pass@1_sd_boot": s,
           "pass@1_by_rollout_index": idx, "pass@1_sd_rollouts": round(sd(idx), 4)}
    for k in ks:
        perk = per_problem_pass_at_k(correct_by_problem, k)
        out[f"pass@{k}"] = None if perk is None else round(mean(perk), 4)
        if perk is not None:
            out[f"pass@{k}_ci95"], out[f"pass@{k}_sd_boot"] = bootstrap(perk, rng)
    return out


def correct_by_problem_from_records(records):
    """{problem_key: [0/1 in rollout-index order]} from rollout records (problem_key, rollout_index, correct)."""
    per = {}
    for r in records:
        per.setdefault(r["problem_key"], []).append((r["rollout_index"], int(bool(r["correct"]))))
    return {pk: [c for _, c in sorted(v)] for pk, v in per.items()}


def scores_from_records(records, ks=KS, seed=SEED):
    """cell_scores over rollout records plus the rates every table prints: cap_rate, committed, tokens.
    Rates are means over rollouts (every problem has the same count once a builder's coverage gate passes)."""
    out = cell_scores(correct_by_problem_from_records(records), ks=ks, seed=seed)
    n = len(records)
    out["cap_rate"] = round(sum(bool(r.get("hit_token_cap")) for r in records) / n, 4)
    out["committed"] = round(sum(bool(r.get("committed")) for r in records) / n, 4)
    toks = [r["output_tokens"] for r in records if r.get("output_tokens") is not None]
    out["mean_output_tokens"] = round(sum(toks) / len(toks), 1) if toks else None
    out["median_output_tokens"] = sorted(toks)[len(toks) // 2] if toks else None
    out["n_problems"] = len({r["problem_key"] for r in records})
    out["n_rollouts"] = n
    return out
