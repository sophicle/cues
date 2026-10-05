#!/usr/bin/env bash
# Create the environment this repo is tested with, then run the CPU unit tests.
#   bash scripts/setup_env.sh                 # -> ./.venv
#   bash scripts/setup_env.sh /scratch/me/env # -> any path (a home directory with a quota is too small: ~25 GB)
# Installs with uv (parallel downloads and unpacking, much faster than pip on network filesystems), bootstrapped
# into the new env. Uses conda for Python 3.11 when python3.11 is not on PATH. Needs Linux x86_64 and an NVIDIA driver that
# supports CUDA 13 (the torch / vLLM wheels bring their own CUDA runtime; no system toolkit is needed).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV=${1:-$REPO/.venv}
if command -v python3.11 >/dev/null; then
  python3.11 -m venv "$ENV"
elif command -v conda >/dev/null; then
  conda create -y -q -p "$ENV" python=3.11 pip
  # A conda env has no bin/activate and carries its own, newer libstdc++. Some nodes' system copy is older, and Python
  # then fails to import (CXXABI_1.3.15 not found). Put the env's lib first, here and in the activate script below.
  CONDA_ENV=1; export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
else
  echo "need python3.11 or conda on PATH"; exit 1
fi
PY=$ENV/bin/python
"$PY" -m pip install -q --upgrade pip uv
# the cache sits beside the env (same filesystem, so uv hardlinks instead of copying; and not in a quota-capped home)
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ENV/.uv-cache}
UV="$PY -m uv pip install --python $PY"
$UV -r "$REPO/requirements.lock"                      # includes deepspeed (32B trainer) and pytest
$UV --no-deps -e "$REPO"
if [ "${CONDA_ENV:-0}" = 1 ]; then   # so ". $ENV/bin/activate" works as it does for a venv
  printf '%s\n' "export PATH=\"$ENV/bin:\$PATH\"" "export LD_LIBRARY_PATH=\"$ENV/lib:\${LD_LIBRARY_PATH:-}\"" "export PYTHON=\"$ENV/bin/python\"" > "$ENV/bin/activate"
fi
"$PY" -m pytest -q "$REPO/tests"
"$PY" -c "import torch, vllm, trl; print('torch', torch.__version__, '| vllm', vllm.__version__, '| trl', trl.__version__, '| cuda available:', torch.cuda.is_available())"
[ "$UV_CACHE_DIR" = "$ENV/.uv-cache" ] && rm -rf "$UV_CACHE_DIR"     # the installed files are hardlinks; they stay
echo
echo "done. Use it with:  export PYTHON=$PY   (the scripts read PYTHON), or  . $ENV/bin/activate"
