#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

datasets=("CommonVoice" "GigaSpeech" "MLS-en" "Emilia-en")
# datasets=("MLS-en")
model_name="gpt-5.4-mini-2026-03-17"
sbert_model="sentence-transformers/all-MiniLM-L6-v2"
sbert_device="cuda:0"
base_dataset="Capspeech"
min_segments="100"
filtered_dataset="${base_dataset}_min${min_segments}"

usage() {
  cat <<'EOF'
Usage: bash 02_get_descriptions.sh [all|combinations|merging|unique|prompts|filter|sbert|summary]

Stages:
  combinations  Build per-dataset profile_combinations.csv files.
  merging       Merge per-dataset profile_combinations.csv into data/Capspeech/.
  unique        Build single-known-field unique_profile_combinations.csv.
  prompts       Build per-dataset profile_prompts.csv and merge into data/Capspeech/.
  filter        Keep profiles with at least N segments in data/Capspeech_minN/.
  sbert         Build SBERT embeddings for the filtered Capspeech directory.
  summary       Print profile/description counts and save exp/tables/profiles.txt.
  all           Run all stages in order.
EOF
}

stage="${1:-all}"
case "$stage" in
  all|combinations|merging|unique|prompts|filter|sbert|summary) ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

run_combinations() {
  for dataset in "${datasets[@]}"; do
    echo "[combinations] ${dataset}"
    srun -p cpu python bin/find_valid_combinations.py \
      --dataset-name "$dataset" \
      --output "data/${dataset}/profile_combinations.csv" &
  done
  wait
}

run_merging() {
  echo "[merging] ${datasets[@]}"
  srun -p cpu python bin/merge_profile_csvs.py profile_combinations.csv --datasets "${datasets[@]}"
}

run_unique() {
  echo "[unique] Capspeech single-known-field profiles"
  srun -p cpu python bin/make_unique_profile_combinations.py \
    --input "data/Capspeech/profile_combinations.csv" \
    --output "data/Capspeech/unique_profile_combinations.csv"
}

run_prompts() {
  source openai_keys.sh
  echo "[prompts] Capspeech"
  srun -p cpu python bin/create_GPT_descriptions.py \
    --input "data/Capspeech/profile_combinations.csv" \
    --output "data/Capspeech/profile_prompts.csv" \
    --model "$model_name" \
    --org "$org_key" \
    --key "$openai_api_key" &

  echo "[prompts] Capspeech unique"
  srun -p cpu python bin/create_GPT_descriptions.py \
    --input "data/Capspeech/unique_profile_combinations.csv" \
    --output "data/Capspeech/unique_profile_prompts.csv" \
    --model "$model_name" \
    --org "$org_key" \
    --key "$openai_api_key" &

  wait

  echo "[prompts] prepend unique prompts to profile prompts"
  tmp_prompts="$(mktemp)"
  {
    cat "data/Capspeech/unique_profile_prompts.csv"
    tail -n +2 "data/Capspeech/profile_prompts.csv"
  } > "$tmp_prompts"
  mv "$tmp_prompts" "data/Capspeech/profile_prompts.csv"

}

run_filter() {
  echo "[filter] ${base_dataset} profiles with num_audios >= ${min_segments} -> ${filtered_dataset}"
  srun -p cpu python bin/filter_capspeech_profiles.py \
    --input-dir "data/${base_dataset}" \
    --output-dir "data/${filtered_dataset}" \
    --min-segments "$min_segments" \
    --overwrite
}

run_sbert() {
  echo "[sbert] ${filtered_dataset}"
  srun -p gpu --gpus 1 python bin/get_SBERT_embeddings.py \
    --input "data/${filtered_dataset}/profile_prompts.csv" \
    --output-dir "exp/SBERT_embs/${filtered_dataset}" \
    --model "$sbert_model" \
    --device "$sbert_device"

}

run_summary() {
  srun -p cpu python bin/profiles_summary.py \
    --datasets "${datasets[@]}" \
    --filtered-datasets "$filtered_dataset"
}

case "$stage" in
    all)
    run_combinations
    run_merging
    run_unique
    run_prompts
    run_filter
    run_sbert
    run_summary
    ;;
  combinations) run_combinations ;;
  merging) run_merging ;;
  unique) run_unique ;;
  prompts) run_prompts ;;
  filter) run_filter ;;
  sbert) run_sbert ;;
  summary) run_summary ;;
esac
