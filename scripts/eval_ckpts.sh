#!/usr/bin/env bash
# Score the checkpoints of an RL run with the same evaluator and grader as the base model (cues eval via
# scripts/eval_grid.sh), one cell per checkpoint x cue.
#   MODEL=olmo7b RUN=$RUNS/rl/olmo7b CKS="base 50 100 150 200 250 300" CUES="none auto" bash scripts/eval_ckpts.sh
#   MODEL=olmo7b RUN=$RUNS/rl/olmo7b_cue CKS="base 300" CUES=auto FINAL=1 BENCH="math500 aime2025" bash scripts/eval_ckpts.sh
#   MODEL=olmo32b RUN=$RUNS/rl/olmo32b MERGE=1 TP=2 bash scripts/eval_ckpts.sh     merge each adapter first: unmerged, a
#                                                                                   32B adapter is 20-30x slower in vLLM
# Two protocols. Default = the trajectory: 8 rollouts per problem at an 8,192-token budget (the paper's training
# curves), cells under $RUNS/eval_8k. FINAL=1 = the paper's table protocol (2 x 16 rollouts, 31,744 tokens), cells
# under $RUNS/eval beside the base cells from scripts/eval_grid.sh, so `python -m cues table` shows base, cue and
# RL together. Labels: base -> <MODEL>, checkpoint N -> <run name>_ck<N>. Merged weights are deleted after
# scoring unless KEEP_MERGED=1. N_ROLLOUTS/BATCHES/BUDGET override either protocol; other knobs pass through to
# eval_grid.sh (BENCH, PROMPT, TP, GPUS, SHARD).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
# A conda-built env carries a newer libstdc++ than some nodes' system copy; let Python find the env's own first.
_envdir="$(cd "$(dirname "$(command -v "$PY")")/.." 2>/dev/null && pwd)" || _envdir=""
[ -n "$_envdir" ] && [ -d "$_envdir/conda-meta" ] && export LD_LIBRARY_PATH="$_envdir/lib:${LD_LIBRARY_PATH:-}"
MODEL=${MODEL:?MODEL=<registry key>}; RUN=${RUN:?RUN=<run directory>}; CKS=${CKS:-"base 50 100 150 200 250 300"}
CUES=${CUES:-"none auto"}; MERGE=${MERGE:-0}; KEEP_MERGED=${KEEP_MERGED:-0}; FINAL=${FINAL:-0}
export RUNS=${RUNS:-$REPO/runs} PYTHONPATH="$REPO:${PYTHONPATH:-}"
if [ "$FINAL" = 1 ]; then OUT=${OUT:-$RUNS/eval}; N_ROLLOUTS=${N_ROLLOUTS:-16}; BATCHES=${BATCHES:-2}; BUDGET=${BUDGET:-31744}
else OUT=${OUT:-$RUNS/eval_8k}; N_ROLLOUTS=${N_ROLLOUTS:-8}; BATCHES=${BATCHES:-1}; BUDGET=${BUDGET:-8192}; fi
NAME=$(basename "$RUN"); FAIL=0
for ck in $CKS; do
  ADAPTER=""; WEIGHTS=""; MERGED=""
  if [ "$ck" = base ]; then LABEL=$MODEL
  else
    LABEL=${NAME}_ck$ck
    [ -f "$RUN/checkpoint-$ck/adapter_config.json" ] || { echo "no checkpoint-$ck under $RUN"; FAIL=1; continue; }
    if [ "$MERGE" = 1 ]; then
      MERGED="$OUT/merged/${NAME}_ck$ck"; mkdir -p "$OUT/logs"
      "$PY" -m cues merge --adapter "$RUN/checkpoint-$ck" --out "$MERGED" --model "$MODEL" > "$OUT/logs/${LABEL}_merge.log" 2>&1 \
        || { echo "merge failed: $OUT/logs/${LABEL}_merge.log"; FAIL=1; continue; }
      WEIGHTS=$MERGED
    else ADAPTER="$RUN/checkpoint-$ck"; fi
  fi
  echo "=== $(date '+%F %T') $LABEL"
  MODEL=$MODEL LABEL=$LABEL ADAPTER=$ADAPTER WEIGHTS=$WEIGHTS CUES=$CUES OUT=$OUT \
    N_ROLLOUTS=$N_ROLLOUTS BATCHES=$BATCHES BUDGET=$BUDGET bash "$REPO/scripts/eval_grid.sh" || FAIL=1
  [ -n "$MERGED" ] && [ "$KEEP_MERGED" != 1 ] && rm -rf "$MERGED"
done
echo "python -m cues table $OUT --benchmark ${BENCH:-math500}"
exit $FAIL
