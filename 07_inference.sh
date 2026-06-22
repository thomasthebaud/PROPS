#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=16
dataset_name="Capspeech_min100"
model='finetune'
train_fraction=0.1
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"

checkpoint="exp/MDN_models/${run_name}.pt"

test_metadata_csv="data/${dataset_name}/test.csv"
test_desc_column="capspeech_desc"
if ! head -1 "$test_metadata_csv" | grep -Eq '(^|,)capspeech_desc(,|$)'; then
  test_desc_column="capspeech_prompt"
fi
output_dir="exp/GMMs/${dataset_name}_test/${run_name}"

echo "Generating one test-set GMM per row from ${test_metadata_csv} using ${checkpoint}"
echo "Using test description column ${test_desc_column}"

srun -p gpu --gpus 1 python bin/inference.py \
  --test-metadata-csv "$test_metadata_csv" \
  --test-desc-column "$test_desc_column" \
  --fallback-profile-csv "data/${dataset_name}/profile_prompts.csv" \
  --device cuda \
  --output-dir "$output_dir" \
  --checkpoint "$checkpoint" \
  --dataset-name "$dataset_name" \
  --gmm-init-path "exp/GMMs/${K}_components_precomputed" \
  --overwrite \
  --model-name ComposedGMM_MDN \
  --batch-size 32

echo "Inference done: ${output_dir}"
