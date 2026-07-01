#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=16
dataset_name="Capspeech_min100"
model='pretrain'
train_fraction=1
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"
gmm_metadata_csv="exp/GMMs/${dataset_name}_test/${run_name}/metadata.csv"
N=1000

mkdir -p "exp/graphs/${run_name}/scatter"

p="unknown"
sr="unknown"
sm="unknown"
g="unknown"
a="unknown"
ac="unknown"

# g="unknown"
# a="teenager,young adult,middle-aged adult,elderly"
# ac="unknown"
# echo "Plotting age scatter for K=${K}"
# srun -p cpu python bin/plot_profile_gmms.py \
#   --test-csv-paths "$dataset_name" \
#   --gmm-metadata-csv "$gmm_metadata_csv" \
#   --gender "$g" \
#   --age "$a" \
#   --accent "$ac" \
#   --pitch "$p" \
#   --speaking-rate "$sr" \
#   --speech-monotony "$sm" \
#   --samples "$N" \
#   --ground-truth-components "$K" \
#   --output "exp/graphs/${run_name}/scatter/K=${K}_${a}_test.png" \
#   --lda &


echo "Plotting gender histogram for K=${K}"
srun -p cpu python bin/plot_gender_histograms.py \
  --test-csv-paths "$dataset_name" \
  --gmm-metadata-csv "$gmm_metadata_csv" \
  --samples "$N" \
  --ground-truth-components "$K" \
  --output "exp/graphs/${run_name}/scatter/K=${K}_male,female_test.png" \
  --lda &

g="unknown"
a="unknown"
ac="american,british,canadian,filipino,indian,irish"
echo "Plotting accent scatter for K=${K}"
srun -p cpu python bin/plot_profile_gmms.py \
  --test-csv-paths "$dataset_name" \
  --gmm-metadata-csv "$gmm_metadata_csv" \
  --gender "$g" \
  --age "$a" \
  --accent "$ac" \
  --pitch "$p" \
  --speaking-rate "$sr" \
  --speech-monotony "$sm" \
  --samples "$N" \
  --ground-truth-components "$K" \
  --output "exp/graphs/${run_name}/scatter/K=${K}_${ac}_test.png" \
  --lda &

wait
echo "Histogram plots written under exp/graphs/${run_name}/scatter"
