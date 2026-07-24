#!/usr/bin/env bash
# Reproduce the streaming 2018 Zhou comparison without exposing a public
# methscope evaluation command. No per-CpG predictions are written.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 DATA.msur ENCODER.upfac RESIDUAL.upres METRICS.tsv [CUDA_DEVICE]" >&2
  exit 1
fi

data=$1
encoder=$2
residual=$3
metrics=$4
device=${5:-0}
methscope_bin=${METHSCOPE:-methscope}

args=(
  _upscale eval
  -i "$data"
  --encoder "$encoder"
  --residual "$residual"
  -o "$metrics"
  --device "$device"
)
if [[ -n ${MAX_CELLS:-} ]]; then
  args+=(--max-cells "$MAX_CELLS")
fi
if [[ -n ${MAX_REPS:-} ]]; then
  args+=(--max-reps "$MAX_REPS")
fi
if [[ -n ${LOG_EVERY_CELLS:-} ]]; then
  args+=(--log-every-cells "$LOG_EVERY_CELLS")
fi

exec "$methscope_bin" "${args[@]}"
