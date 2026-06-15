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
datasets=(CommonVoice GigaSpeech MLS-en Emilia-en)
datasets=(CommonVoice)


#To kill : squeue -h -u "$USER" -o "%i %j" | awk '$2 ~ /^gs/ {print $1}' | xargs -r scancel


for dataset_name in "${datasets[@]}"; do

  for split in "${splits[@]}"; do
    metadata_csv="data/$dataset_name/$split.csv"

    output_dataset_name="${dataset_name}_${split}"
    split_num_gpus="$num_gpus"
    if [[ "$split" == "dev" || "$split" == "test" ]]; then
      split_num_gpus="2"
    fi
    echo "Submitting xvector extraction for ${output_dataset_name} (${split_num_gpus} shard(s))"

    for ((shard = 0; shard < split_num_gpus; shard++)); do
      srun -p gpu --gpus 1 --job-name "cv${shard}_${split}" python bin/extract_xvectors.py \
        --dataset-name "$output_dataset_name" \
        --metadata-csv "$metadata_csv" \
        --data-root "$data_root" \
        --exp-root "$exp_root" \
        --model-name "$model_name" \
        --source "$source_model" \
        --device "$device" \
        --num-shards "$split_num_gpus" \
        --shard-index "$shard" \
        --fail-stop &
    done
  done
done

wait
echo "Xvector extraction completed for datasets ${datasets[*]} and splits ${splits[*]}."

# datasets=(CommonVoice GigaSpeech Emilia-en)
# for dataset in "${datasets[@]}"; do
#   srun -p cpu -c 16 python bin/check_xvectors_exists.py "${datasets[@]}" --num-workers 16
# done
