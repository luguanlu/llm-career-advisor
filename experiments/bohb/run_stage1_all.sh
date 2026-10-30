#!/usr/bin/env bash
set -euo pipefail

# Batch runner for BOHB stage-1 configs
# Usage:
#  - Default: runs `llamafactory-cli train --config <cfg>` for each candidate in background
#      ./experiments/bohb/run_stage1_all.sh
#  - Use Accelerate (distributed) by setting environment variables:
#      USE_ACCELERATE=1 ACCEL_CONFIG=examples/accelerate/fsdp2_config.yaml NUM_PROCESSES=4 ./experiments/bohb/run_stage1_all.sh

BASEDIR=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG_DIR="$BASEDIR/experiments/bohb"
LOGDIR="$CONFIG_DIR/logs"
mkdir -p "$LOGDIR"

ACCEL=${USE_ACCELERATE:-0}
ACCEL_CONFIG=${ACCEL_CONFIG:-examples/accelerate/fsdp2_config.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-4}

echo "Starting BOHB stage-1 batch runner"
echo "Config dir: $CONFIG_DIR"
echo "Logs: $LOGDIR"

for cfg in "$CONFIG_DIR"/bohb_stage1_*.yaml; do
  [ -e "$cfg" ] || continue
  name=$(basename "$cfg" .yaml)
  log="$LOGDIR/${name}.log"
  echo "Launching $name -> $log"
  if [ "$ACCEL" = "1" ]; then
    accelerate launch --config_file "$ACCEL_CONFIG" --num_machines 1 --num_processes "$NUM_PROCESSES" src/train.py "$cfg" >"$log" 2>&1 &
  else
    nohup "$CONFIG_DIR/llamafactory-cli" train --config "$cfg" >"$log" 2>&1 &
  fi
  # small delay to avoid instant overload
  sleep 2
done

echo "All jobs launched (background). Monitor logs in $LOGDIR"
