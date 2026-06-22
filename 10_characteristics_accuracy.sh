#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=16
dataset_name="Capspeech_min100"
model='finetune'
train_fraction=0.1
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"
gmm_metadata_csv="exp/GMMs/${dataset_name}_test/${run_name}/metadata.csv"
N=1000
min_labels=100
ordinal_classifier_mode="svr"
classifier_train_dataset="${dataset_name}_dev"
test_dataset="${dataset_name}_test"
output_dir="exp/characteristics_prediction/${run_name}_${min_labels}"
per_characteristic_dir="${output_dir}/per_characteristic"
characteristics=(gender age accent speech_monotony pitch speaking_rate)

mkdir -p "$output_dir" "$per_characteristic_dir"

echo "Evaluating characteristic SVM accuracies with one CPU task per characteristic"
echo "Ordinal classifier mode: ${ordinal_classifier_mode}"

pids=()
input_csvs=()
for characteristic in "${characteristics[@]}"; do
  characteristic_dir="${per_characteristic_dir}/${characteristic}"
  characteristic_csv="${characteristic_dir}/accuracies.csv"
  mkdir -p "$characteristic_dir"
  input_csvs+=("$characteristic_csv")

  echo "Launching ${characteristic} classifier evaluation"
  srun -p cpu -c 4 python bin/characteristics_accuracy.py \
    --classifier-train-csv-paths "$classifier_train_dataset" \
    --test-csv-paths "$test_dataset" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gmm-metadata-csv "$gmm_metadata_csv" \
    --output-csv "$characteristic_csv" \
    --confusion-output-dir "$output_dir" \
    --characteristics "$characteristic" \
    --ordinal-classifier-mode "$ordinal_classifier_mode" \
    --min-label-xvectors $min_labels \
    --num-load-workers 4 \
    --samples "$N" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Merging per-characteristic results"
python bin/merge_characteristics_accuracy.py \
  --input-csvs "${input_csvs[@]}" \
  --output-csv "${output_dir}/accuracies.csv" \
  --results-tex "${output_dir}/results.tex"

echo "Wrote ${output_dir}/accuracies.csv and ${output_dir}/results.tex"
