#!/bin/bash
set -euo pipefail

# ====== Datasets / Methods / Ranking weight ======
configs=("metabric")
methods=("clisurv-po" "clisurv-ph" "clisurv-gen")
w_ranks=(0 0.05 0.1 0.2 0.5 0.7 1.0)   # change as needed

# Optional: choose GPU visibility once here
VISIBLE_GPUS="1"

for config in "${configs[@]}"; do
  for method in "${methods[@]}"; do
    for w_rank in "${w_ranks[@]}"; do

      echo "[RUN] cfg=${config} method=${method} w_rank=${w_rank}"

      if ! python3 main.py \
        --debug \
        --config "${config}" \
        --method "${method}" \
        --w_rank "${w_rank}" \
        --visible_gpus "${VISIBLE_GPUS}"; then
        echo "❌ Failed run (but continuing)."
      fi

    done
  done
done