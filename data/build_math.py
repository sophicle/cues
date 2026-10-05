"""Build the RL training pool from the MATH train split, held out from MATH-500.

    python data/build_math.py            # writes data/math_train.jsonl and data/CHECKSUMS

Sources (the original hendrycks/competition_math is delisted; these are the live mirrors):
  EleutherAI/hendrycks_math   seven subject configs, train split = 7,500 problems
  HuggingFaceH4/MATH-500      the 500-problem held-out subset, used only to flag overlap

gt is the last brace-balanced \\boxed{...} in the reference solution. A problem whose solution has
no box cannot be graded and is dropped; the count is printed. The paper's runs used an earlier
version of this builder whose regex missed nested braces and kept 7,393 problems.

Fields: {problem_key, capability, problem, gt, level, split, in_math500}; problem_key is the
subject slug plus a hash of the normalized problem text (the same keys as data/problems/math500.jsonl).

Provenance: avdravid/reasoning_registers_grpo data/build_pool.py.
"""
import hashlib
import json
import re
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

HERE = Path(__file__).resolve().parent
POOL_SRC = "EleutherAI/hendrycks_math"
HELD_SRC = "HuggingFaceH4/MATH-500"


def boxed(text):
    start = text.rfind("\\boxed{")
    while start != -1:
        i, depth = start + len("\\boxed{"), 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if not depth:
                    return text[start + len("\\boxed{"):i].strip()
            i += 1
        start = text.rfind("\\boxed{", 0, start)
    return None


def norm_text(s):
    return re.sub(r"\s+", " ", s).strip()


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def key(subject, problem):
    return f"{slug(subject)}__{hashlib.sha1(norm_text(problem).encode()).hexdigest()[:6]}"


def level_int(lv):
    m = re.search(r"\d+", str(lv)) if lv is not None else None
    return int(m.group(0)) if m else None


def main():
    held = load_dataset(HELD_SRC, split="test")
    held_norm = {norm_text(r["problem"]) for r in held}
    rows, seen, dropped, overlap = [], set(), 0, 0
    for cfg in get_dataset_config_names(POOL_SRC):
        for r in load_dataset(POOL_SRC, cfg, split="train"):
            gt = boxed(r["solution"])
            if gt is None or gt == "":
                dropped += 1
                continue
            subject = r.get("type", cfg)
            pk = key(subject, r["problem"])
            if pk in seen:
                continue
            seen.add(pk)
            in500 = norm_text(r["problem"]) in held_norm
            overlap += in500
            if in500:
                continue
            rows.append({"problem_key": pk, "capability": slug(subject), "problem": r["problem"], "gt": gt,
                         "level": level_int(r.get("level")), "split": "train", "in_math500": False})
    out = HERE / "math_train.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (HERE / "CHECKSUMS").write_text(f"{sha}  math_train.jsonl\n")
    print(f"wrote {out}: {len(rows)} problems (dropped {dropped} with no boxed answer, {overlap} overlapping MATH-500)")
    by_cap = {}
    for r in rows:
        by_cap[r["capability"]] = by_cap.get(r["capability"], 0) + 1
    for cap, n in sorted(by_cap.items()):
        print(f"  {cap:24s} {n}")
    print(f"sha256 {sha}")


if __name__ == "__main__":
    main()
