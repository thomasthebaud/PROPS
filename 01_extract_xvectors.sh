#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_name="ecapa_tdnn"
source_model="speechbrain/spkrec-ecapa-voxceleb"
data_root="data"
exp_root="exp/xvectors"
device="cuda"
num_gpus="10"

if [[ "${1:-}" == "--num-gpus" ]]; then
  num_gpus="${2:-}"
  shift 2
fi

if [[ $# -gt 0 ]] || ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 [--num-gpus N]" >&2
  exit 2
fi

splits=(test train dev)
splits=(test)
datasets=(CommonVoice GigaSpeech MLS-en Emilia-en)
datasets=(Emilia-en)


for dataset_name in "${datasets[@]}"; do

  for split in "${splits[@]}"; do
    metadata_csv="data/$dataset_name/$split.csv"

    output_dataset_name="${dataset_name}_${split}"
    echo "Submitting xvector extraction for ${output_dataset_name} (${num_gpus} shard(s))"

    for ((shard = 0; shard < num_gpus; shard++)); do
      srun -p gpu --gpus 1 python bin/extract_xvectors.py \
        --dataset-name "$output_dataset_name" \
        --metadata-csv "$metadata_csv" \
        --data-root "$data_root" \
        --exp-root "$exp_root" \
        --model-name "$model_name" \
        --source "$source_model" \
        --device "$device" \
        --num-shards "$num_gpus" \
        --shard-index "$shard" &
    done

    wait
  done
done
