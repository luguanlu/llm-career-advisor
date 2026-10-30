#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

# Use a single free GPU for sequential runs.
export CUDA_VISIBLE_DEVICES=3

for i in 02 03 04 05 06 07 08 09 10 11 12; do
  yaml="experiments/bohb/bohb_stage1_${i}.yaml"
  log="experiments/bohb/logs/bohb_stage1_${i}_sequential.log"
  echo "=== START ${yaml} at $(date) ===" | tee "$log"
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate llama_env
  python3 -m llamafactory.cli train "$yaml" 2>&1 | tee -a "$log"
  ret=${PIPESTATUS[0]}
  echo "=== END ${yaml} at $(date) EXIT=${ret} ===" | tee -a "$log"
  if [ "$ret" -ne 0 ]; then
    echo "[WARNING] ${yaml} failed with exit code ${ret}" | tee -a "$log"
  fi
  echo
  # Add a small pause to let CUDA cleanup before next run.
  sleep 5

done
