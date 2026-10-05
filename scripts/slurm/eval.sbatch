#!/bin/bash
# Slurm template for an evaluation or a cue search on one node: runs the given command from the repo root
# with the submitting shell's environment (the knobs go through --export).
#   sbatch --export=ALL,MODEL=olmo7b,BENCH="math500 gsm8k" scripts/slurm/eval.sbatch scripts/eval_grid.sh
#   sbatch --export=ALL,MODEL=olmo7b,RUN=$RUNS/rl/olmo7b,CKS="base 100 200 300" scripts/slurm/eval.sbatch scripts/eval_ckpts.sh
#   sbatch --gres=gpu:1 scripts/slurm/eval.sbatch scripts/search.sh --model qwen14b --prompt boxed
#SBATCH --job-name=token_cues_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
exec bash "$@"
