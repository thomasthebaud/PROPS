#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=1
dataset_name="Capspeech_min100"
model='finetune'
train_fraction=0.01
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"

mkdir -p "exp/graphs/${run_name}/scatter"

p="unknown"
sr="unknown"
sm="unknown"

for GT_K in 4; do
  g="unknown"
  a="teenager,young adult,middle-aged adult,elderly"
  ac="unknown"
  echo "Plotting age/gender scatter for GT_K=${GT_K}"
  srun -p cpu python bin/plot_profile_gmms.py \
    --test-csv-paths "$dataset_name" \
    --gmm-metadata-csv "exp/GMMs/${run_name}_desc0/metadata.csv" \
    --gender "$g" \
    --age "$a" \
    --accent "$ac" \
    --pitch "$p" \
    --speaking-rate "$sr" \
    --speech-monotony "$sm" \
    --ground-truth-components "$GT_K" \
    --output "exp/graphs/${run_name}/scatter/K=${GT_K}_${a}_test.png" \
    --lda &

  g="male,female"
  a="unknown"
  ac="american"
  echo "Plotting accent/gender scatter for GT_K=${GT_K}"
  srun -p cpu python bin/plot_profile_gmms.py \
    --test-csv-paths "$dataset_name" \
    --gmm-metadata-csv "exp/GMMs/${run_name}_desc0/metadata.csv" \
    --gender "$g" \
    --age "$a" \
    --accent "$ac" \
    --pitch "$p" \
    --speaking-rate "$sr" \
    --speech-monotony "$sm" \
    --ground-truth-components "$GT_K" \
    --output "exp/graphs/${run_name}/scatter/K=${GT_K}_${g}_test.png" \
    --lda &
done

wait
echo "Scatter plots written under exp/graphs/${run_name}/scatter"
