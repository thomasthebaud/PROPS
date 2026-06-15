#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=1
dataset_name="Capspeech_min100"
train_fraction=0.1
pretrain_fraction=1
run_name="${K}_components_${dataset_name}_p=${train_fraction}"
pretrain_checkpoint="exp/MDN_models/${K}_components_${dataset_name}_p=${pretrain_fraction}_pretrain.pt"
output_checkpoint="exp/MDN_models/${run_name}_finetune.pt"

echo "Fine-tuning ComposedGMM_MDN from ${pretrain_checkpoint} using per-sample Capspeech descriptions"

srun -p gpu --gpus 1 python bin/finetune.py \
  --epochs 10 \
  --batch-size 256 \
  --device cuda \
  --checkpoint "$pretrain_checkpoint" \
  --output "$output_checkpoint" \
  --dataset-name "$dataset_name" \
  --train-fraction "$train_fraction" \
  --train-csv-paths "data/${dataset_name}/train.csv" \
  --dev-csv-paths "data/${dataset_name}/dev.csv" \
  --desc-column auto \
  --gmm-init-path "exp/GMMs/${K}_components_precomputed" \
  --entropy-weight 0 \
  --mean-norm-weight 0 \
  --variance-down-weight 0 \
  --lr 0.00001
