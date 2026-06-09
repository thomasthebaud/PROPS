#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_name="${MODEL_NAME:-ecapa_tdnn}"
source_model="${SOURCE_MODEL:-speechbrain/spkrec-ecapa-voxceleb}"
data_root="${DATA_ROOT:-data}"
exp_root="${EXP_ROOT:-exp/xvectors}"
device="${DEVICE:-cuda}"

datasets=("CommonVoice_dev" "CommonVoice_test" "GigaSpeech_test" "CommonVoice_train" "GigaSpeech_train")



for dataset_name in "${datasets[@]}"; do
  echo "Submitting xvector extraction for ${dataset_name}"
  srun -p gpu --gpus 1 python bin/extract_xvectors.py \
    --dataset-name "$dataset_name" \
    --data-root "$data_root" \
    --exp-root "$exp_root" \
    --model-name "$model_name" \
    --source "$source_model" \
    --device "$device"
done
