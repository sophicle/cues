#!/bin/bash
# Slurm template for a training run: edit the account/partition/time lines for your cluster and submit from
# the repo root. Every attempt resumes from the newest complete checkpoint and exits at once when the run is
# finished, so allocations can be chained with --dependency=afterany:<jobid>.
#   sbatch scripts/slurm/train.sbatch --model olmo7b --out $RUNS/rl/olmo7b
#   sbatch scripts/slurm/train.sbatch --model olmo7b --out $RUNS/rl/olmo7b_cue --cue auto
#   sbatch --nodes=2 scripts/slurm/train.sbatch --model olmo32b --out $RUNS/rl/olmo32b --zero3 --vllm-tp 8 --per-device 1
#SBATCH --job-name=token_cues_rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
# sbatch runs a spooled copy of this file, so $0 may not locate the repo; fall back to the submit dir
REPO="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)"
[ -x "$REPO/scripts/train.sh" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -x "$REPO/scripts/train.sh" ] || { echo "cannot find scripts/train.sh: submit from the repo root"; exit 1; }
OUT=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && OUT=$a; prev=$a; done
[ -n "$OUT" ] || { echo "pass --out <dir>"; exit 1; }
[ -f "$OUT/DONE" ] && { echo "$OUT/DONE exists: the run is finished, nothing to do"; exit 0; }
"$REPO/scripts/slurm/quarantine.sh" "$OUT"
RESUME=""; ls -d "$OUT"/checkpoint-* 2>/dev/null | grep -qE '/checkpoint-[0-9]+$' && RESUME="--resume"
# On Savio's A100 nodes the first NCCL all-reduce hangs with GPU peer-to-peer on (all ranks wait until the
# 30-minute watchdog kills them); route through host memory instead. Set NCCL_P2P_DISABLE=0 where P2P works.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NNODES=$SLURM_JOB_NUM_NODES MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  bash -c 'NODE_RANK=$SLURM_PROCID exec "$0" "$@"' "$REPO/scripts/train.sh" "$@" $RESUME
