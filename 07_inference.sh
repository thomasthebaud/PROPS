#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=4
dataset_name="Capspeech_min100"
model='pretrain'
train_fraction=1
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"

checkpoint="exp/MDN_models/${run_name}.pt"

echo "Generating desc0 profile GMMs from ${checkpoint}"

for desc in "desc0" "capspeech_desc"; do

  srun -p gpu --gpus 1 python bin/inference.py \
    --test-on-profile $desc \
    --device cuda \
    --output-dir "exp/GMMs/${run_name}_desc0" \
    --checkpoint "$checkpoint" \
    --dataset-name "$dataset_name" \
    --gmm-init-path "exp/GMMs/${K}_components_precomputed" \
    --model-name ComposedGMM_MDN &

done

wait

echo "All inference done"
