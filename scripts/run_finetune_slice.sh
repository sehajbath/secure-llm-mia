#!/bin/bash
# Launch finetuning for a given slice on Colab or local machine.
# Usage: ./scripts/run_finetune_slice.sh <slice_id> <replay_tag> <seed>
set -euo pipefail

SLICE_ID=${1:-1}
REPLAY_TAG=${2:-noreplay}
SEED=${3:-17}

echo "[TODO] Trigger notebook execution for slice ${SLICE_ID} (${REPLAY_TAG}) with seed ${SEED}."
echo "Consider using papermill or colab-cli once paths are configured."
