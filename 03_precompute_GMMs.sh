#!/usr/bin/env bash
set -euo pipefail

K=256
num_shards=10
num_workers=16
output_dir="exp/GMMs/${K}_components_precomputed"
job_group_id="GMM"
dataset_name="Capspeech_min100"

echo "Precomputing one K=${K} GMM per profile from data/${dataset_name}/train.csv using ${num_shards} shards"
echo "Using job name prefix ${job_group_id}"

for ((shard = 0; shard < num_shards; shard++)); do
  shard_job_name="${job_group_id}_$((shard + 1))"
  srun -p cpu -c $num_workers --job-name "$shard_job_name" python bin/precompute_GMMs.py \
    --train-csv-paths "data/${dataset_name}/train.csv" \
    --data-root data \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --profile-combinations-csv data/${dataset_name}/profile_prompts.csv \
    --output-dir "$output_dir" \
    --K ${K} \
    --num-load-workers "$num_workers" \
    --num-profile-workers "$num_workers" \
    --num-shards "$num_shards" \
    --shard-index "$shard" \
    --max-xvectors-per-profile 3500 \
    "$@" &
    
    sleep 1
done

# #To kill : squeue -h -u "$USER" -o "%i %j" | awk '$2 ~ /^GMM_/ {print $1}' | xargs -r scancel
wait

python - <<MERGEPY
from pathlib import Path

output_dir = Path("$output_dir")
for name in ["metadata", "skipped_profiles"]:
    shard_paths = [output_dir / f"{name}.shard{idx}.csv" for idx in range($num_shards)]
    existing = [path for path in shard_paths if path.exists()]
    if not existing:
        continue
    output_path = output_dir / f"{name}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as out:
        wrote_header = False
        for path in existing:
            with path.open(encoding="utf-8", newline="") as src:
                header = src.readline()
                if header and not wrote_header:
                    out.write(header)
                    wrote_header = True
                for line in src:
                    out.write(line)
    print(f"Wrote {output_path} from {len(existing)} shard files", flush=True)
MERGEPY
