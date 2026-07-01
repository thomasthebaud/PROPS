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
svc_c_values=(0.01 0.1 1 10 100 1000 10000)
ordinal_classifier_mode="svr"
classifier_train_dataset="${dataset_name}_dev"
test_dataset="${dataset_name}_test"
characteristics=(gender age accent speech_monotony pitch speaking_rate)
workers=16

echo "Evaluating characteristic SVM accuracies with one CPU task per characteristic"
echo "Ordinal classifier mode: ${ordinal_classifier_mode}"
for c_value in "${svc_c_values[@]}"; do
  output_dir="exp/characteristics_prediction/${run_name}_${min_labels}/C=${c_value}"
  per_characteristic_dir="${output_dir}/per_characteristic"
  mkdir -p "$output_dir" "$per_characteristic_dir"

  echo "Evaluating SVC C=${c_value}"
  pids=()
  input_csvs=()
  for characteristic in "${characteristics[@]}"; do
    characteristic_dir="${per_characteristic_dir}/${characteristic}"
    characteristic_csv="${characteristic_dir}/accuracies.csv"
    mkdir -p "$characteristic_dir"
    input_csvs+=("$characteristic_csv")

    echo "Launching ${characteristic} classifier evaluation with SVC C=${c_value}"
    srun -p cpu -c $workers python bin/characteristics_accuracy.py \
      --classifier-train-csv-paths "$classifier_train_dataset" \
      --test-csv-paths "$test_dataset" \
      --xvector-root exp/xvectors/ecapa_tdnn \
      --gmm-metadata-csv "$gmm_metadata_csv" \
      --output-csv "$characteristic_csv" \
      --confusion-output-dir "$output_dir" \
      --characteristics "$characteristic" \
      --ordinal-classifier-mode "$ordinal_classifier_mode" \
      --svc-c "$c_value" \
      --min-label-xvectors $min_labels \
      --num-load-workers $workers \
      --samples "$N" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "Merging per-characteristic results for SVC C=${c_value}"
  python bin/merge_characteristics_accuracy.py \
    --input-csvs "${input_csvs[@]}" \
    --output-csv "${output_dir}/accuracies.csv" \
    --results-tex "${output_dir}/results.tex"

  echo "Wrote ${output_dir}/accuracies.csv and ${output_dir}/results.tex" 
done

echo "Writing per-metric C comparison CSVs"
python bin/compare_characteristics_c_metrics.py \
  --base-dir "exp/characteristics_prediction/${run_name}_${min_labels}" \
  --c-values "${svc_c_values[@]}"
