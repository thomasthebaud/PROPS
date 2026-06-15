#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=1
dataset_name="Capspeech_min100"
model='pretrain'
desc='capspeech_desc'
desc='desc0'
train_fraction=1
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"
N=1000
classifier_train_dataset="${dataset_name}_dev"
test_dataset="${dataset_name}_test"

echo "Evaluating characteristic SVM accuracies"
mkdir -p "exp/characteristics_prediction/${run_name}"

srun -p cpu python bin/characteristics_accuracy.py \
  --classifier-train-csv-paths "$classifier_train_dataset" \
  --test-csv-paths "$test_dataset" \
  --xvector-root exp/xvectors/ecapa_tdnn \
  --gmm-metadata-csv "exp/GMMs/${run_name}_${desc}/metadata.csv" \
  --output-csv "exp/characteristics_prediction/${run_name}/accuracies_${desc}.csv" \
  --characteristics gender,age,accent,speech_monotony,pitch,speaking_rate \
  --min-label-xvectors 25 \
  --samples "$N"
