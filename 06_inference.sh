#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=4
CV_p=1

checkpoint="exp/MDN_models/${K}_components_CV=${CV_p}.pt"
output_dir="exp/GMMs/${K}_components_CV=${CV_p}_mixed"

echo "Generating desc0 profile GMMs from ${checkpoint}"
mkdir -p "$output_dir"

srun -p gpu --gpus 1 python bin/inference.py \
  --test-on-profile "desc0" \
  --device cuda \
  --output-dir "$output_dir" \
  --checkpoint "$checkpoint" \
  --overwrite \
  --model-name ComposedGMM_MDN
