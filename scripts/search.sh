#!/usr/bin/env bash
# The cue search for one model; the screen is spread over the node's visible GPUs (CUDA_VISIBLE_DEVICES or
# --gpus N to restrict; TP=2 groups them in pairs for a 32B).
#   bash scripts/search.sh --model qwen14b --prompt boxed
#   TP=2 bash scripts/search.sh --model olmo32b
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
# A conda-built env carries a newer libstdc++ than some nodes' system copy; let Python find the env's own first.
_envdir="$(cd "$(dirname "$(command -v "$PY")")/.." 2>/dev/null && pwd)" || _envdir=""
[ -n "$_envdir" ] && [ -d "$_envdir/conda-meta" ] && export LD_LIBRARY_PATH="$_envdir/lib:${LD_LIBRARY_PATH:-}"
export RUNS=${RUNS:-$REPO/runs} PYTHONPATH="$REPO:${PYTHONPATH:-}" VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/$USER/tri_search} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/ind_search}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exec "$PY" -m cues search --tensor-parallel-size "${TP:-1}" "$@"
