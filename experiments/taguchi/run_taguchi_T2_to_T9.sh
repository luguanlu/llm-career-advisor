#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=3

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate llama_env

mkdir -p experiments/taguchi/logs

for id in T2 T3 T4 T5 T6 T7 T8 T9; do
  yaml="experiments/taguchi/configs/taguchi_${id}.yaml"
  log="experiments/taguchi/logs/taguchi_${id}.log"

  echo "=== START ${yaml} at $(date) ===" | tee "$log"
  python3 -m llamafactory.cli train "$yaml" 2>&1 | tee -a "$log"
  ret=${PIPESTATUS[0]}
  echo "=== END ${yaml} at $(date) EXIT=${ret} ===" | tee -a "$log"

  if [ "$ret" -ne 0 ]; then
    echo "[WARNING] ${yaml} failed with exit code ${ret}" | tee -a "$log"
  fi

  sleep 5
done
