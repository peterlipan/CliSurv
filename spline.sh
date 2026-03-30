#!/bin/bash
set -euo pipefail

# ====== Datasets / Methods / Ranking weight ======
configs=("gbsg")
methods=("clisurv-po" "clisurv-ph" "clisurv-gen")
z_ms=(2 4 8 10 12 14)   # change as needed

# Optional: choose GPU visibility once here
VISIBLE_GPUS="1"

for config in "${configs[@]}"; do
  for method in "${methods[@]}"; do
    for z_m in "${z_ms[@]}"; do

      echo "[RUN] cfg=${config} method=${method} z_m=${z_m}"

      if ! python3 main.py \
        --debug \
        --config "${config}" \
        --method "${method}" \
        --z_m "${z_m}" \
        --visible_gpus "${VISIBLE_GPUS}"; then
        echo "❌ Failed run (but continuing)."
      fi

    done
  done
done