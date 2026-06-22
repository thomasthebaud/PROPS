#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

dataset_name="Capspeech"
profile_counts_dataset="Capspeech"

Ks="1,2,4,8,16,32,64,128,256,512,1024,2048"
K_values=(1 2 4 8 16 32 64 128 256 512 1024 2048)
num_shards=10
num_workers=16
precompute_gmm_root="exp/GMMs/stage08_precomputed"
profile_tmp_root="exp/GMMs/stage08_profile_subsets"
job_group_id="GMM08"

eval_gmm_root="$precompute_gmm_root"
output_dir="exp/graphs/Best_K"
cache_dir="exp/Best_K"

run_Precompute=0
run_Evaluate=0

if [[ $# -eq 0 ]]; then
  run_Precompute=1
  run_Evaluate=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --precompute|--stage1)
      run_Precompute=1
      ;;
    --evaluate|--stage2)
      run_Evaluate=1
      ;;
    --all)
      run_Precompute=1
      run_Evaluate=1
      ;;
    -h|--help)
      cat <<HELP
Usage: bash 08_test_precomputed_GMMs.sh [--precompute] [--evaluate] [--all]

Stages:
  --precompute   Stage 1: precompute GMMs for K=${Ks} into ${precompute_gmm_root}
  --evaluate     Stage 2: evaluate precomputed GMMs on ${dataset_name} dev/test xvectors

Stage 1 filters profiles to num_audios >= 2*K and uses max_xvectors_per_profile=100*K.
It waits for all shards of one K to finish before launching the next K.

If no stage is passed, all stages are run.
HELP
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

make_profile_subset() {
  local k="$1"
  local min_audios=$((2 * k))
  local profile_subset="$profile_tmp_root/profile_prompts_min_${min_audios}_for_K_${k}.csv"
  mkdir -p "$profile_tmp_root"
  python - <<PY
import csv
from pathlib import Path

src = Path("data/${dataset_name}/profile_prompts.csv")
dst = Path("$profile_subset")
min_audios = int("$min_audios")
with src.open(newline="", encoding="utf-8") as f, dst.open("w", newline="", encoding="utf-8") as out:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
    writer.writeheader()
    kept = 0
    for row in reader:
        try:
            num_audios = int(float(row.get("num_audios", 0)))
        except ValueError:
            num_audios = 0
        if num_audios >= min_audios:
            writer.writerow(row)
            kept += 1
print(f"Wrote {dst} with {kept} profiles having num_audios >= {min_audios}", file=__import__("sys").stderr, flush=True)
PY
  echo "$profile_subset"
}

merge_precompute_shards() {
  local output_dir="$1"
  local num_shards_for_k="$2"
  python - <<PY
from pathlib import Path

output_dir = Path("$output_dir")
for name in ["metadata", "skipped_profiles"]:
    shard_paths = [output_dir / f"{name}.shard{idx}.csv" for idx in range($num_shards_for_k)]
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
PY
}

if [[ "$run_Precompute" -eq 1 ]]; then
  echo "Stage 1: precomputing GMMs for K=${Ks} into ${precompute_gmm_root}"

  for K in "${K_values[@]}"; do
    max_xvectors_per_profile=$((100 * K))
    min_audios=$((2 * K))
    precompute_output_dir="${precompute_gmm_root}/${K}_components_precomputed"
    profile_subset="$(make_profile_subset "$K")"

    echo "Precomputing K=${K}: profiles with num_audios >= ${min_audios}, max_xvectors_per_profile=${max_xvectors_per_profile}"
    mkdir -p "$precompute_output_dir"

    for ((shard = 0; shard < num_shards; shard++)); do
      shard_job_name="${job_group_id}_K${K}_$((shard + 1))"
      srun -p cpu -c "$num_workers" --job-name "$shard_job_name" python bin/precompute_GMMs.py \
        --train-csv-paths "data/${dataset_name}/train.csv" \
        --data-root data \
        --xvector-root exp/xvectors/ecapa_tdnn \
        --profile-combinations-csv "$profile_subset" \
        --output-dir "$precompute_output_dir" \
        --K "$K" \
        --num-load-workers "$num_workers" \
        --num-profile-workers "$num_workers" \
        --num-shards "$num_shards" \
        --shard-index "$shard" \
        --max-xvectors-per-profile "$max_xvectors_per_profile" &

      sleep 1
    done

    wait
    merge_precompute_shards "$precompute_output_dir" "$num_shards"
    echo "Finished K=${K}"
  done
fi

if [[ "$run_Evaluate" -eq 1 ]]; then
  echo "Stage 2: evaluating precomputed GMMs for K=${Ks} on ${dataset_name} dev/test xvectors"
  mkdir -p "$output_dir" "$cache_dir"

  srun -p cpu python bin/test_precomputed_gmms.py \
    --dataset-name "$dataset_name" \
    --ks "$Ks" \
    --gmm-root "$eval_gmm_root" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --output-dir "$output_dir" \
    --cache-dir "$cache_dir" \
    --profile-counts-csv "data/${profile_counts_dataset}/profile_combinations.csv"

  echo "Wrote Best-K NLL table and plot under ${output_dir}"
fi
