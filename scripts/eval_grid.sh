#!/usr/bin/env bash
# The paper's evaluation protocol for one model: every (benchmark x cue) cell, sharded over this node's
# GPUs, 32 rollouts per problem as two seeded batches of 16, then summarized.
#   MODEL=olmo7b BENCH="math500 gsm8k" CUES="none auto" bash scripts/eval_grid.sh
#   MODEL=qwen14b PROMPT=boxed BENCH=aime2025 bash scripts/eval_grid.sh
#   MODEL=olmo32b TP=2 bash scripts/eval_grid.sh                                    a 32B on 80 GB cards
#   MODEL=olmo7b_think EXTRA="--chat" CUES=none bash scripts/eval_grid.sh           a chat model through its template
#   MODEL=olmo7b ADAPTER=$RUNS/rl/olmo7b/checkpoint-300 LABEL=olmo7b_ck300 BUDGET=7168 bash scripts/eval_grid.sh
# Knobs (defaults): PROMPT (the registry's), CUES ("none auto"; a literal cue with spaces or newlines has to
# go through `python -m cues eval --cues $'...'` directly), N_ROLLOUTS=16 per batch, BATCHES=2, SEED=20260819
# (rollout index r is sampled with SEED + r, so the second batch continues the first), BUDGET=31744 (7168 for 8,192-context checkpoints), SHARD=25 problems per vLLM task,
# TP=1, GPUS (all visible), OUT=$RUNS/eval, WEIGHTS (merged checkpoint), FAMILY (for a Hub id), EXTRA.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
# A conda-built env carries a newer libstdc++ than some nodes' system copy; let Python find the env's own first.
_envdir="$(cd "$(dirname "$(command -v "$PY")")/.." 2>/dev/null && pwd)" || _envdir=""
[ -n "$_envdir" ] && [ -d "$_envdir/conda-meta" ] && export LD_LIBRARY_PATH="$_envdir/lib:${LD_LIBRARY_PATH:-}"
MODEL=${MODEL:?MODEL=<registry key | Hub id | directory>}
BENCH=${BENCH:-math500}; CUES=${CUES:-"none auto"}; PROMPT=${PROMPT-}; FAMILY=${FAMILY-}; EXTRA=${EXTRA-}
N_ROLLOUTS=${N_ROLLOUTS:-16}; BATCHES=${BATCHES:-2}; BUDGET=${BUDGET:-31744}; SHARD=${SHARD:-25}; TP=${TP:-1}
export RUNS=${RUNS:-$REPO/runs}; OUT=${OUT:-$RUNS/eval}; ADAPTER=${ADAPTER-}; WEIGHTS=${WEIGHTS-}
LABEL=${LABEL:-$(basename "$MODEL")${ADAPTER:+_$(basename "$ADAPTER")}}
SEED=${SEED:-20260819}   # rollout index r is sampled with SEED + r (cues/evaluate.py), so batches never repeat
export PYTHONPATH="$REPO:${PYTHONPATH:-}" VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
if [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  IFS=, read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
else
  DEVICES=()
  while IFS= read -r device; do DEVICES+=("$device"); done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
NG=${GPUS:-${#DEVICES[@]}}
[[ "$NG" =~ ^[1-9][0-9]*$ && "$TP" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS and TP must be positive integers"; exit 1; }
[ "$NG" -le "${#DEVICES[@]}" ] || { echo "GPUS=$NG exceeds visible devices"; exit 1; }
WORKERS=$((NG / TP)); [ "$WORKERS" -ge 1 ] || { echo "TP=$TP exceeds the $NG GPUs"; exit 1; }
LOGS=$OUT/logs; mkdir -p "$LOGS"
COMMON=(--model "$MODEL" --label "$LABEL" --max-new-tokens "$BUDGET" --n-rollouts "$N_ROLLOUTS" --out "$OUT" --tensor-parallel-size "$TP")
[ -n "$PROMPT" ] && COMMON+=(--prompt "$PROMPT")
[ -n "$FAMILY" ] && COMMON+=(--family "$FAMILY")
[ -n "$ADAPTER" ] && COMMON+=(--adapter "$ADAPTER")
[ -n "$WEIGHTS" ] && COMMON+=(--weights "$WEIGHTS")

# one task per (benchmark, problem shard, batch)
TASKS=()
for b in $BENCH; do
  f=$REPO/data/problems/$b.jsonl
  [ -f "$f" ] || f=$b
  [ -f "$f" ] || { echo "no such benchmark: $b"; exit 1; }
  n=$(grep -c . "$f")
  for ((s = 0; s < n; s += SHARD)); do
    for ((k = 0; k < BATCHES; k++)); do TASKS+=("$b $s $k"); done
  done
done
echo "$(date '+%F %T') $LABEL: ${#TASKS[@]} tasks over $WORKERS worker(s) x $TP GPU(s); logs in $LOGS"

run_task() {   # worker, benchmark, problem start, batch
  local w=$1 b=$2 s=$3 k=$4 gpus="" suffix=""
  for ((g = w * TP; g < (w + 1) * TP; g++)); do gpus+="${gpus:+,}${DEVICES[$g]}"; done
  [ "$k" -gt 0 ] && suffix="_b$((k + 1))"
  local bn; bn=$(basename "${b%.jsonl}")
  local log="$LOGS/${LABEL}_${bn}_p${s}${suffix}.log"
  CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=/tmp/$USER/tri_$w TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/ind_$w VLLM_CACHE_ROOT=/tmp/$USER/vllm_$w \
    "$PY" -m cues eval "${COMMON[@]}" --benchmark "$b" --problem-start "$s" --problem-count "$SHARD" \
      --seed "$SEED" --rollout-index-offset $((k * N_ROLLOUTS)) --suffix "$suffix" --cues $CUES $EXTRA > "$log" 2>&1 \
    || { echo "FAILED: $log"; return 1; }
}
worker() {     # worker w takes tasks w, w+WORKERS, ...: the shards are the same size, so round robin balances
  local w=$1 failed=0
  for ((i = w; i < ${#TASKS[@]}; i += WORKERS)); do run_task "$w" ${TASKS[$i]} || failed=1; done
  return "$failed"
}
PIDS=()
for ((w = 0; w < WORKERS; w++)); do worker "$w" & PIDS+=("$!"); done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
[ "$FAILED" -eq 0 ] || { echo "Evaluation failed; see task logs. No summary produced."; exit 1; }

case " $BENCH " in *" humaneval "*) "$PY" -m cues exec-grade --rollouts-glob "$OUT/$LABEL/*/humaneval/*/rollouts_p*.jsonl";; esac
"$PY" -m cues summarize "$OUT" --label "$LABEL" --rollouts $((N_ROLLOUTS * BATCHES))
echo "$(date '+%F %T') done: python -m cues table $OUT --benchmark <benchmark>"
