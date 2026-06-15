#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=1
dataset_name="Capspeech_min100"
model='pretrain'
train_fraction=1
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"
N_min=10

mkdir -p "exp/graphs/${run_name}"

for category in age gender accent; do
  echo "Computing ${category} log-likelihood confusion matrix"
  srun -p cpu python bin/graphs/confusion_matrix.py \
    --category "$category" \
    --test-csv-paths $dataset_name \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gmm-metadata-csv "exp/GMMs/${run_name}_desc0/metadata.csv" \
    --output-dir "exp/graphs/${run_name}" \
    --n-min "$N_min"
done

echo "Confusion matrices written under exp/graphs/${run_name}"
