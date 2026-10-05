"""Grade HumanEval rollouts by running the tests, in an isolated Docker container per program.

    python -m cues exec-grade                                    # every humaneval cell under $RUNS/eval
    python -m cues exec-grade --rollouts-glob '$RUNS/eval/olmo7b/*/humaneval/*/rollouts_p*.jsonl'

For every record: map problem_key -> HumanEval task (prompt, test, entry_point); pull the candidate
code out of the completion (the first ```python block that defines the entry point, else the first
python block, else any fenced block, else the raw text; a bare function body gets the HumanEval prompt
prepended); run prompt + code + test + check(entry_point) with a read-only filesystem, resource limits and no network; write
the record plus exec_correct / exec_error / exec_extract to a sibling directory <cell>_execgraded/.
Input files are never touched. cues.summarize scores HumanEval cells from the graded sibling only.

Provenance: sophicle/reason scripts/eval/exec_grade_humaneval.py.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid

from cues.evaluate import DATA, RUNS, read_jsonl

PROBLEMS_DEFAULT = str(DATA / "humaneval.jsonl")
TIMEOUT_S = 10.0
MEM_BYTES = 2 * 1024 ** 3



def load_humaneval() -> dict[str, dict]:
    """{task_id: {prompt, test, entry_point, canonical_solution}} from the Hub (cached under HF_HOME)."""
    from datasets import load_dataset
    return {r["task_id"]: r for r in load_dataset("openai/openai_humaneval", split="test")}


def map_problem(problem_key: str, problem_text: str | None, tasks: dict[str, dict]) -> dict | None:
    if problem_key in tasks:
        return tasks[problem_key]
    if problem_text:                       # fall back: the entry point named in the prompt text
        for t in tasks.values():
            if f"def {t['entry_point']}(" in problem_text:
                return t
    return None



FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)
KEEP_TOPLEVEL = ("def ", "import ", "from ", "class ", "@", "#", "async def ", ")", "]", "}")


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).lower(), m.group(2)) for m in FENCE_RE.finditer(text)]


def strip_trailing_prose(code: str) -> str:
    """Cut the code at the first top-level line that is not code, once a def body has started."""
    out, in_body = [], False
    for ln in code.split("\n"):
        stripped = ln.strip()
        if not stripped or ln[0] in " \t":
            out.append(ln)
            continue
        if stripped.startswith(KEEP_TOPLEVEL) or stripped.startswith(("if __name__", "print(", "assert ")) \
                or re.match(r"^[\w.\[\]]+\s*(=|\+=|-=|\()", stripped):
            if stripped.startswith(("def ", "async def ", "class ")):
                in_body = True
            out.append(ln)
            continue
        if in_body:
            break
    return "\n".join(out).rstrip() + "\n"


def extract_code(completion: str, task: dict) -> tuple[str, str]:
    """(program_code, how): how records the extraction path."""
    entry = task["entry_point"]
    blocks = fenced_blocks(completion)
    py = [b for lang, b in blocks if lang in ("python", "py", "python3")]
    code = how = None
    for b in py:
        if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", b, re.M):
            code, how = b, "fence_python_entry"
            break
    if code is None and py:
        code, how = py[0], "fence_python_first"
    if code is None and blocks:
        for lang, b in blocks:
            if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", b, re.M):
                code, how = b, "fence_any_entry"
                break
        if code is None:
            code, how = blocks[0][1], "fence_any_first"
    if code is None:
        code, how = completion, "raw"

    code = code.replace(" ", "\n").replace(" ", "\n")
    m = re.search(rf"def\s+{re.escape(entry)}\s*\(", code)     # "Answer: def foo(...):" -- prose before the def
    if m:
        line_start = code.rfind("\n", 0, m.start()) + 1
        if code[line_start:m.start()].strip():
            code = code[:line_start] + code[m.start():]
    has_def = re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, re.M) is not None
    if not has_def:                        # a function body: indent if needed, prepend the HumanEval prompt
        body_lines = code.split("\n")
        first = next((ln for ln in body_lines if ln.strip()), "")
        if first and first[0] not in " \t":
            body_lines = [("    " + ln) if ln.strip() else ln for ln in body_lines]
        body = strip_trailing_prose("\n".join(body_lines))
        code = task["prompt"].rstrip("\n") + "\n" + body
        how += "+prompt_prepended"
    else:                                  # drop prose before the first top-level code line, normalise indentation
        lines = code.split("\n")
        start = next((i for i, ln in enumerate(lines)
                      if re.match(r"^\s*(def |async def |from |import |class |@)", ln)), 0)
        lines = lines[start:]
        lines[0] = lines[0].lstrip()
        code = strip_trailing_prose(textwrap.dedent("\n".join(lines)))
    return code, how



GUARD = r'''
import builtins as _b, os as _os, sys as _sys, faulthandler as _fh
_fh.disable()
try:
    import socket as _s
    _s.socket = None; _s.create_connection = None; _s.getaddrinfo = None
except Exception:
    pass
for _n in ("kill","system","putenv","remove","removedirs","rmdir","fchdir","setuid","fork","forkpty",
           "killpg","rename","renames","truncate","replace","unlink","fchmod","fchown","chmod","chown",
           "chroot","lchown","chdir"):
    if hasattr(_os, _n): setattr(_os, _n, None)
try:
    import shutil as _sh; _sh.rmtree = None; _sh.move = None; _sh.chown = None
except Exception:
    pass
try:
    import subprocess as _sp; _sp.Popen = None
except Exception:
    pass
_sys.modules["ipdb"] = None; _sys.modules["joblib"] = None; _sys.modules["psutil"] = None
_sys.modules["tkinter"] = None; _sys.modules["resource"] = None
_b.exit = None; _b.quit = None; _b.help = None
'''


def build_program(task: dict, code: str) -> str:
    return (task["prompt"].rstrip("\n") + "\n\n" + code.rstrip("\n") + "\n\n"
            + task["test"].rstrip("\n") + f"\n\ncheck({task['entry_point']})\n")


SANDBOX_IMAGE = os.environ.get("CUES_SANDBOX_IMAGE", "python:3.11-slim")


def require_sandbox():
    """Fail closed: never execute generated programs directly on the host."""
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=30)
        subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE], check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit("HumanEval requires Docker and a locally available sandbox image. "
                         f"Pull {SANDBOX_IMAGE} first (or set CUES_SANDBOX_IMAGE). No code was executed.") from exc


def sandbox_command(path, name, python):
    return ["docker", "run", "--rm", "--pull=never", "--name", name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--user=65534:65534",
            "--pids-limit=64", "--memory=2g", "--memory-swap=2g", "--cpus=1",
            "--ulimit", "nofile=64:64", "--ulimit", "fsize=16777216:16777216",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
            "--mount", f"type=bind,src={path},dst=/program.py,readonly",
            "--workdir=/tmp", "--env=PYTHONDONTWRITEBYTECODE=1",
            SANDBOX_IMAGE, python, "-I", "-B", "/program.py"]


def run_program(program: str, python: str = "python3", timeout: float = TIMEOUT_S) -> tuple[bool, str]:
    name = "cues-he-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="he_") as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(GUARD + "\n" + program)
        os.chmod(path, 0o444)
        try:
            # Discard output so generated programs cannot exhaust host memory through a pipe.
            p = subprocess.run(sandbox_command(path, name, python),
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=timeout)
            if p.returncode in (125, 126, 127):
                raise RuntimeError(f"HumanEval container failed to start (exit {p.returncode}); grading aborted")
            return (True, "") if p.returncode == 0 else (False, f"exit {p.returncode}")
        except subprocess.TimeoutExpired:
            return False, "timeout"
        finally:
            subprocess.run(["docker", "rm", "--force", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)




def out_path_for(in_path: str) -> str:
    d, name = os.path.split(os.path.abspath(in_path))
    return os.path.join(d.rstrip("/") + "_execgraded", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts-glob", nargs="+", default=[str(RUNS / "eval" / "*" / "*" / "humaneval" / "*" / "rollouts_p*.jsonl")],
                    help="glob(s) for rollouts_p*.jsonl; default: every humaneval cell under $RUNS/eval")
    ap.add_argument("--problems-file", default=PROBLEMS_DEFAULT)
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--python", default="python3", help="interpreter inside the sandbox image")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--overwrite", action="store_true", help="regrade even if the output file exists")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N records per file")
    args = ap.parse_args()

    files = sorted({f for g in args.rollouts_glob for f in glob.glob(g)})
    files = [f for f in files if ".tmp" not in os.path.basename(f) and not os.path.abspath(f).split(os.sep)[-2].endswith("_execgraded")]
    if not files:
        sys.exit("no rollout files matched")
    require_sandbox()
    tasks = load_humaneval()
    problems = {r["problem_key"]: r for r in read_jsonl(args.problems_file)} if os.path.exists(args.problems_file) else {}
    print(f"[exec_grade] {len(files)} files, {len(tasks)} HumanEval tasks, workers={args.workers}, "
          f"sandbox={SANDBOX_IMAGE}", file=sys.stderr)

    records: list[tuple[str, dict, str | None, str]] = []
    jobs: dict[str, str] = {}
    for f in files:
        if not args.overwrite and os.path.exists(out_path_for(f)):
            print(f"[exec_grade] skip (exists): {out_path_for(f)}", file=sys.stderr)
            continue
        recs = read_jsonl(f)
        if args.limit:
            recs = recs[:args.limit]
        for r in recs:
            task = map_problem(r["problem_key"], problems.get(r["problem_key"], {}).get("problem"), tasks)
            if task is None:
                records.append((f, r, None, "no_task"))
                continue
            code, how = extract_code(r.get("completion") or "", task)
            program = build_program(task, code)
            key = hashlib.sha1((task["task_id"] + "\0" + program).encode("utf-8", "surrogatepass")).hexdigest()
            jobs.setdefault(key, program)
            records.append((f, r, key, how))
    print(f"[exec_grade] {len(records)} records, {len(jobs)} unique programs", file=sys.stderr)

    results: dict[str, tuple[bool, str]] = {}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_program, prog, args.python, args.timeout): k for k, prog in jobs.items()}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            if i % 500 == 0 or i == len(futs):
                print(f"[exec_grade] {i}/{len(futs)} programs run ({time.time() - t0:.0f}s)", file=sys.stderr)

    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for f, r, key, how in records:
        r = dict(r)
        if key is None:
            r["exec_correct"], r["exec_error"] = False, "no_task_for_problem_key"
        else:
            r["exec_correct"], r["exec_error"] = results[key]
        r["exec_extract"] = how
        by_file[f].append(r)
    for f, recs in by_file.items():
        op = out_path_for(f)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        with open(op + ".tmp", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(op + ".tmp", op)
        print(f"[exec_grade] wrote {op}", file=sys.stderr)

    summary: dict[str, dict] = {}
    for f in files:
        op = out_path_for(f)
        if not os.path.exists(op):
            continue
        cell = os.path.relpath(os.path.dirname(os.path.abspath(f)), str(RUNS / "eval"))
        for r in read_jsonl(op):
            s = summary.setdefault(cell, {"keys": set(), "n": 0, "exec": 0, "stored": 0, "cap": 0,
                                          "agree": 0, "err": collections.Counter()})
            s["keys"].add(r["problem_key"]); s["n"] += 1
            s["exec"] += bool(r["exec_correct"]); s["stored"] += bool(r.get("correct"))
            s["cap"] += bool(r.get("hit_token_cap")); s["agree"] += bool(r["exec_correct"]) == bool(r.get("correct"))
            if not r["exec_correct"]:
                e = r["exec_error"]
                s["err"]["timeout" if e == "timeout" else e.split(":")[0][:40]] += 1
    print()
    print(f"{'cell':<60} {'n_prob':>6} {'n_roll':>6} {'pass@1_exec':>11} {'pass@1_stored':>13} {'cap_rate':>8} {'agree':>6}")
    for cell, s in sorted(summary.items()):
        n = s["n"]
        print(f"{cell[-60:]:<60} {len(s['keys']):>6} {n:>6} {s['exec'] / n:>11.4f} {s['stored'] / n:>13.4f} "
              f"{s['cap'] / n:>8.4f} {s['agree'] / n:>6.3f}")
    print()
    for cell, s in sorted(summary.items()):
        print(f"  {cell}: top errors: " + ", ".join(f"{k}={v}" for k, v in s["err"].most_common(4)))


if __name__ == "__main__":
    main()
