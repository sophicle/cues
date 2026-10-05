#!/usr/bin/env bash
# Node-level smoke (2-8 GPUs, ~1 hour): the sharded two-batch evaluation protocol, the HumanEval execution
# grader, a chat model through its template, a multi-GPU GRPO run with checkpoints, resume and the DONE guard,
# the no-penalty variant, and checkpoint scoring served as an adapter and as merged weights. Needs the
# pinned environment and the models in the Hub cache.
#   bash tests/smoke_node.sh            # writes under $RUNS/smoke_node, wiped first
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
# A conda-built env carries a newer libstdc++ than some nodes' system copy; let Python find the env's own first.
_envdir="$(cd "$(dirname "$(command -v "$PY")")/.." 2>/dev/null && pwd)" || _envdir=""
[ -n "$_envdir" ] && [ -d "$_envdir/conda-meta" ] && export LD_LIBRARY_PATH="$_envdir/lib:${LD_LIBRARY_PATH:-}"
export RUNS=${RUNS:-$REPO/runs} PYTHONPATH="$REPO:${PYTHONPATH:-}" VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
NG=$(nvidia-smi -L | wc -l); [ "$NG" -ge 2 ] || { echo "needs at least 2 GPUs"; exit 1; }
OUT=$RUNS/smoke_node; rm -rf "$OUT"; mkdir -p "$OUT"
step() { echo; echo "=== $(date '+%T') $*"; }
count() { "$PY" -c "import sys; print(sum(1 for l in open(sys.argv[1]) if l.strip()))" "$1"; }

step "a 10-problem MATH-500 subset for the evaluation steps"
head -10 "$REPO/data/problems/math500.jsonl" > "$OUT/math10.jsonl"

step "eval_grid.sh: qwen4b, two seeded batches of 2 rollouts, shards of 5 problems, no cue and the cue"
MODEL=qwen4b BENCH=$OUT/math10.jsonl CUES="none auto" N_ROLLOUTS=2 BATCHES=2 SHARD=5 BUDGET=512 OUT=$OUT/eval bash "$REPO/scripts/eval_grid.sh"
for slug in none spTospdetermine; do
  cell=$OUT/eval/qwen4b/boxed/math10/$slug
  for f in rollouts_p00000.jsonl rollouts_p00000_b2.jsonl rollouts_p00005.jsonl rollouts_p00005_b2.jsonl; do test -f "$cell/$f"; done
  "$PY" - "$cell" <<'PYEOF'
import json, sys, pathlib, glob
cell = pathlib.Path(sys.argv[1])
recs = [json.loads(l) for f in glob.glob(str(cell / "rollouts_p*.jsonl")) for l in open(f) if l.strip()]
assert len(recs) == 40, len(recs)
assert {r["rollout_index"] for r in recs} == {0, 1, 2, 3}, "second batch did not offset its rollout index"
assert len({r["problem_key"] for r in recs}) == 10
by = {}
for r in recs: by.setdefault(r["problem_key"], {})[r["rollout_index"]] = r["completion"]
dup = sum(d[1] == d[2] for d in by.values())          # last of batch 1 vs first of batch 2
assert dup <= 2, f"batch 2 repeats batch 1 on {dup}/10 problems: the second batch's seeds overlap the first's"
s = json.load(open(cell / "summary.json")); assert s["n_rollouts"] == 40 and s["rollouts_per_problem"] == [4, 4]
print(f"  {cell.name}: 4 shards, 40 records, pass@1 {s['pass@1']}")
PYEOF
done
"$PY" -m cues table "$OUT/eval" --benchmark math10

step "eval_grid.sh: HumanEval, 1 rollout per problem, graded by execution"
MODEL=qwen4b BENCH=humaneval CUES=none N_ROLLOUTS=1 BATCHES=1 SHARD=41 BUDGET=512 OUT=$OUT/eval bash "$REPO/scripts/eval_grid.sh"
test -d "$OUT/eval/qwen4b/boxed/humaneval/none_execgraded"
"$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); assert s['graded_by']=='execution' and s['n_rollouts']==164, s; print('  humaneval pass@1 by execution', s['pass@1'])" "$OUT/eval/qwen4b/boxed/humaneval/none/summary.json"

step "a chat model through its template: olmo7b_instruct, 4 problems, no cue"
"$PY" -m cues eval --model olmo7b_instruct --benchmark math500 --problem-count 4 --n-rollouts 1 --max-new-tokens 512 --cues none --out "$OUT/eval"
test -f "$OUT/eval/olmo7b_instruct/chat/math500/none/rollouts_p00000.jsonl"

step "train: olmo7b on all $NG GPUs, 4 steps x 8 prompts, a checkpoint every 2 steps, no cue, truncation penalty"
bash "$REPO/scripts/train.sh" --model olmo7b --out "$OUT/rl/olmo7b" --steps 4 --save-steps 2 --prompts-per-step 8 --per-device 1 --gpu-mem 0.30
test -f "$OUT/rl/olmo7b/checkpoint-2/adapter_config.json" && test -f "$OUT/rl/olmo7b/checkpoint-4/adapter_config.json" && test -f "$OUT/rl/olmo7b/DONE"

step "resume to step 6 (DONE removed), then the DONE guard"
rm "$OUT/rl/olmo7b/DONE"
bash "$REPO/scripts/train.sh" --model olmo7b --out "$OUT/rl/olmo7b" --steps 6 --save-steps 2 --prompts-per-step 8 --per-device 1 --gpu-mem 0.30 --resume
test -f "$OUT/rl/olmo7b/checkpoint-6/adapter_config.json" && test -f "$OUT/rl/olmo7b/DONE"
"$PY" -m cues train --model olmo7b --out "$OUT/rl/olmo7b" --steps 6 --resume | grep -q "DONE exists"
"$PY" -m cues openers "$OUT/rl/olmo7b" --window 2
"$PY" - "$OUT/rl/olmo7b" <<'PYEOF'
import json, sys, glob
recs = [json.loads(l) for f in glob.glob(sys.argv[1] + "/rollout_log/rollouts_rank*.jsonl") for l in open(f) if l.strip()]
ranks = len(glob.glob(sys.argv[1] + "/rollout_log/rollouts_rank*.jsonl"))
assert all(r["reward"] == 0.0 for r in recs if r["capped"]), "a capped rollout was not scored 0"
print(f"  {len(recs)} rollouts logged by {ranks} ranks, {sum(r['capped'] for r in recs)} capped (all scored 0), register share {sum(r['register'] for r in recs)/len(recs):.2f}")
PYEOF

step "the no-penalty variant: qwen4b, 2 steps"
bash "$REPO/scripts/train.sh" --model qwen4b --out "$OUT/rl/qwen4b_nopen" --steps 2 --save-steps 1000 --prompts-per-step 8 --per-device 1 --gpu-mem 0.30 --no-trunc-penalty
"$PY" - "$OUT/rl/qwen4b_nopen" <<'PYEOF'
import json, sys, glob
assert json.load(open(sys.argv[1] + "/run_args.json"))["trunc_penalty"] is False
recs = [json.loads(l) for f in glob.glob(sys.argv[1] + "/rollout_log/rollouts_rank*.jsonl") for l in open(f) if l.strip()]
capped = [r for r in recs if r["capped"]]
assert all(r["reward"] is None for r in capped), "a capped rollout kept a reward under --no-trunc-penalty"
print(f"  {len(recs)} rollouts, {len(capped)} capped -> reward None (dropped)")
PYEOF

step "eval_ckpts.sh: base + checkpoint-2 as an adapter, then checkpoint-4 merged, trajectory protocol on the subset"
MODEL=olmo7b RUN=$OUT/rl/olmo7b CKS="base 2" CUES="none auto" BENCH=$OUT/math10.jsonl N_ROLLOUTS=2 BUDGET=512 SHARD=5 OUT=$OUT/eval_8k bash "$REPO/scripts/eval_ckpts.sh"
MODEL=olmo7b RUN=$OUT/rl/olmo7b CKS="4" CUES=none MERGE=1 KEEP_MERGED=1 BENCH=$OUT/math10.jsonl N_ROLLOUTS=2 BUDGET=512 SHARD=5 OUT=$OUT/eval_8k bash "$REPO/scripts/eval_ckpts.sh"
for cell in olmo7b/rlzero/math10/none olmo7b/rlzero/math10/p2Okay olmo7b_ck2/rlzero/math10/none olmo7b_ck2/rlzero/math10/p2Okay olmo7b_ck4/rlzero/math10/none; do test -f "$OUT/eval_8k/$cell/summary.json"; done
test -f "$OUT/eval_8k/merged/olmo7b_ck4/config.json"
"$PY" -m cues table "$OUT/eval_8k" --benchmark math10

step "node smoke passed"
