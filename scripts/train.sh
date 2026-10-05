#!/usr/bin/env bash
# accelerate launch wrapper for the trainer. Run on the allocated node(s); for several nodes set NNODES and
# MASTER_ADDR on each and give every node a different NODE_RANK (scripts/slurm/train.sbatch does this).
#   scripts/train.sh --model olmo7b --out $RUNS/rl/olmo7b
#   scripts/train.sh --model olmo7b --out $RUNS/rl/olmo7b_cue --cue auto
#   scripts/train.sh --model qwen14b --out $RUNS/rl/qwen14b --per-device 1 --gpu-mem 0.45
#   NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<head> scripts/train.sh --model olmo32b --out $RUNS/rl/olmo32b --zero3 --vllm-tp 8 --per-device 1
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
# A conda-built env carries a newer libstdc++ than some nodes' system copy; let Python find the env's own first.
_envdir="$(cd "$(dirname "$(command -v "$PY")")/.." 2>/dev/null && pwd)" || _envdir=""
[ -n "$_envdir" ] && [ -d "$_envdir/conda-meta" ] && export LD_LIBRARY_PATH="$_envdir/lib:${LD_LIBRARY_PATH:-}"
NNODES="${NNODES:-1}"; NODE_RANK="${NODE_RANK:-0}"
GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"; MASTER_PORT="${MASTER_PORT:-29611}"
export RUNS=${RUNS:-$REPO/runs} PYTHONPATH="$REPO:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
# node-local compile caches: N ranks compiling into one network directory fails with stale handles
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton_$$}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/inductor_$$}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
# DeepSpeed (--zero3) refuses to import without a CUDA toolkit; the pip-installed one inside the env is enough
NVCU=$(ls -d "$("$PY" -c 'import sys; print(sys.prefix)')"/lib/python3*/site-packages/nvidia/cu13 2>/dev/null | head -1)
[ -z "${CUDA_HOME:-}" ] && [ -n "$NVCU" ] && export CUDA_HOME="$NVCU"
export DS_SKIP_CUDA_CHECK=1
# expandable_segments conflicts with the allocator vLLM sleep mode uses; only set it with --no-sleep
case " $* " in *" --no-sleep "*) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;; esac
exec "$PY" -m accelerate.commands.launch --num_machines "$NNODES" --num_processes $((NNODES * GPUS)) \
  --machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  "$REPO/cues/rl/train.py" "$@"
