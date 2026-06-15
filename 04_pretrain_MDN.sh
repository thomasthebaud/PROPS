#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=4
dataset_name="Capspeech_min100"
train_fraction=1
run_name="${K}_components_${dataset_name}_p=${train_fraction}"

echo "Pretraining ComposedGMM_MDN pi routing with ${K} components"

srun -p gpu --gpus 1 python bin/pretrain.py \
  --epochs 100 \
  --batch-size 512 \
  --hidden-dims '1024,2048,1024' \
  --num-components "$K" \
  --device cuda \
  --output "exp/MDN_models/${run_name}_pretrain.pt" \
  --dataset-name "$dataset_name" \
  --profile-csv-path "data/${dataset_name}/profile_prompts.csv" \
  --sbert-root "exp/SBERT_embs/${dataset_name}" \
  --gmm-init-path "exp/GMMs/${K}_components_precomputed" \
  --pi-profile-ce-weight 1.0 \
  --lr 0.0001
