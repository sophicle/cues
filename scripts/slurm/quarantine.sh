#!/usr/bin/env bash
# Move aside any checkpoint cut mid-write, so --resume picks the newest complete one. Complete means
# trainer_state.json plus optimizer state: optimizer.pt, or a global_step*/ directory under ZeRO-3.
#   scripts/slurm/quarantine.sh <out_dir>
OUT=${1:?usage: quarantine.sh <out_dir>}
for c in "$OUT"/checkpoint-*/; do
  [ -d "$c" ] || continue
  case "$c" in *.partial_*) continue;; esac
  if [ -f "${c}trainer_state.json" ] && { [ -f "${c}optimizer.pt" ] || ls -d "${c}"global_step* >/dev/null 2>&1; }; then continue; fi
  echo "quarantining partial checkpoint $c"
  mv "$c" "${c%/}.partial_$(date +%s)"
done
