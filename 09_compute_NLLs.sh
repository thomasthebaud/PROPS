#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=16
dataset_name="Capspeech_min100"
model_name="ComposedGMM_MDN"
gmm_init_path="exp/GMMs/${K}_components_precomputed"
batch_size=128

run_PretrainGMMs=0
run_FinetuneGMMs=0
run_NLLScores=0
run_RandomScores=0
run_Table=0

if [[ $# -eq 0 ]]; then
  run_PretrainGMMs=1
  run_FinetuneGMMs=1
  run_NLLScores=1
  run_RandomScores=1
  run_Table=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrain|--stage1)
      run_PretrainGMMs=1
      ;;
    --finetune|--stage2)
      run_FinetuneGMMs=1
      ;;
    --scores|--stage3)
      run_NLLScores=1
      ;;
    --random|--stage4)
      run_RandomScores=1
      ;;
    --table|--stage5)
      run_Table=1
      ;;
    --all)
      run_PretrainGMMs=1
      run_FinetuneGMMs=1
      run_NLLScores=1
      run_RandomScores=1
      run_Table=1
      ;;
    -h|--help)
      cat <<EOF
Usage: bash 09_compute_NLLs.sh [--pretrain] [--finetune] [--scores] [--random] [--table] [--all]

Stages:
  --pretrain   Stage 1: compute pretrained-model GMMs
  --finetune   Stage 2: compute finetuned-model GMMs
  --scores     Stage 3: compute dev/test NLL score CSVs
  --random     Stage 4: score GMMs against random unit vectors
  --table      Stage 5: summarize NLL scores in a LaTeX table

If no stage is passed, all stages are run.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

run_inference_job() {
  local model="$1"
  local train_fraction="$2"
  local split="$3"
  local desc_column="$4"

  local run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"
  local checkpoint="exp/MDN_models/${run_name}.pt"
  local metadata_csv="data/${dataset_name}/${split}.csv"
  local output_dir="exp/GMMs/${dataset_name}_${split}/${run_name}_${desc_column}"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    return 1
  fi
  if [[ ! -f "$metadata_csv" ]]; then
    echo "Missing metadata CSV: ${metadata_csv}" >&2
    return 1
  fi

  echo "Generating ${model} p=${train_fraction} ${split} ${desc_column} GMMs into ${output_dir}"
  srun -p gpu --gpus 1 python bin/inference.py \
    --test-metadata-csv "$metadata_csv" \
    --test-desc-column "$desc_column" \
    --fallback-profile-csv "data/${dataset_name}/profile_prompts.csv" \
    --device cuda \
    --output-dir "$output_dir" \
    --checkpoint "$checkpoint" \
    --dataset-name "$dataset_name" \
    --gmm-init-path "$gmm_init_path" \
    --model-name "$model_name" \
    --batch-size "$batch_size"
}

if [[ "$run_PretrainGMMs" -eq 1 ]]; then
  echo "Stage 1: pretrained ${model_name} GMM inference"

  run_inference_job "pretrain" "1" "test" "desc0" &
  run_inference_job "pretrain" "1" "test" "capspeech_prompt" &
  run_inference_job "pretrain" "1" "dev" "desc1" &
  run_inference_job "pretrain" "1" "dev" "capspeech_prompt" &

  wait
fi

if [[ "$run_FinetuneGMMs" -eq 1 ]]; then
  echo "Stage 2: finetuned ${model_name} GMM inference"

  run_inference_job "finetune" "0.1" "test" "desc0" &
  run_inference_job "finetune" "0.1" "test" "capspeech_prompt" &
  run_inference_job "finetune" "0.1" "dev" "desc1" &
  run_inference_job "finetune" "0.1" "dev" "capspeech_prompt" &

  wait
fi

if [[ "$run_NLLScores" -eq 1 ]]; then
  echo "Stage 3: computing NLL score CSVs"

  mkdir -p exp/NLL_scores/jobs
  for model_variant in pretrain pretrain_uniform finetune; do
    case "$model_variant" in
      pretrain)
        model=pretrain
        train_fraction=1
        weighting=learned
        ;;
      pretrain_uniform)
        model=pretrain
        train_fraction=1
        weighting=uniform_pi
        ;;
      finetune)
        model=finetune
        train_fraction=0.1
        weighting=learned
        ;;
    esac

    for eval_set in dev test; do
      if [[ "$eval_set" == "dev" ]]; then
        gmm_split=dev
        desc_columns=(desc1 capspeech_prompt)
      else
        gmm_split=test
        desc_columns=(desc0 capspeech_prompt)
      fi

      for desc_column in "${desc_columns[@]}"; do
        output_csv="exp/NLL_scores/jobs/${model_variant}_${eval_set}_${desc_column}.csv"
        srun -p cpu python bin/compute_nll_scores.py \
          --dataset-name "$dataset_name" \
          --k "$K" \
          --output-root exp/NLL_scores \
          --model "$model" \
          --train-fraction "$train_fraction" \
          --gmm-split "$gmm_split" \
          --eval-set "$eval_set" \
          --desc-column "$desc_column" \
          --weighting "$weighting" \
          --output-csv "$output_csv" \
          --overwrite &
      done
    done
  done

  wait
fi

if [[ "$run_RandomScores" -eq 1 ]]; then
  echo "Stage 4: scoring dev/test xvectors against random baselines"

  mkdir -p exp/NLL_scores/random
  for eval_set in dev test; do
    for baseline_model in normal random; do
      output_csv="exp/NLL_scores/random/${baseline_model}_${eval_set}.csv"
      srun -p cpu python bin/compute_nll_scores_to_random.py         --dataset-name "$dataset_name"         --k "$K"         --output-root exp/NLL_scores         --split "$eval_set"         --model "$baseline_model"         --output-csv "$output_csv"         --overwrite &
    done
  done

  wait
fi

if [[ "$run_Table" -eq 1 ]]; then
  echo "Stage 5: aggregating NLL table"

  srun -p cpu python bin/make_nll_table.py \
    --scores-root exp/NLL_scores \
    --output-path exp/NLL_scores/table.tex \
    --digits 1

  cat exp/NLL_scores/table.tex
  cp exp/NLL_scores/table.tex exp/tables/nll_scores.tex
fi

echo "Done. Wrote NLL scores under exp/NLL_scores and LaTeX table to exp/NLL_scores/table.tex"
