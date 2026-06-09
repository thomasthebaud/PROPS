#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=4
CV_p=1

mkdir -p "exp/graphs/${K}_components_CV=${CV_p}/scatter"

p="unknown"
sr="unknown"
sm="unknown"

for GT_K in 1 4; do
  g="male,female"
  a="twenties,thirties,fourties"
  ac="unknown"
  echo "Plotting age/gender scatter for GT_K=${GT_K}"
  srun -p cpu python bin/plot_profile_gmms.py \
    --test-csv-paths "CommonVoice_test" \
    --gmm-metadata-csv "exp/GMMs/${K}_components_CV=${CV_p}_mixed/metadata.csv" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gender "$g" \
    --age "$a" \
    --accent "$ac" \
    --pitch "$p" \
    --speaking-rate "$sr" \
    --speech-monotony "$sm" \
    --ground-truth-components "$GT_K" \
    --output "exp/graphs/${K}_components_CV=${CV_p}/scatter/K=${GT_K}_${g}_${a}_test.png" \
    --lda &

  g="male,female"
  a="unknown"
  ac="US,England"
  echo "Plotting accent/gender scatter for GT_K=${GT_K}"
  srun -p cpu python bin/plot_profile_gmms.py \
    --test-csv-paths "CommonVoice_test" \
    --gmm-metadata-csv "exp/GMMs/${K}_components_CV=${CV_p}_mixed/metadata.csv" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gender "$g" \
    --age "$a" \
    --accent "$ac" \
    --pitch "$p" \
    --speaking-rate "$sr" \
    --speech-monotony "$sm" \
    --ground-truth-components "$GT_K" \
    --output "exp/graphs/${K}_components_CV=${CV_p}/scatter/K=${GT_K}_${g}_${ac}_test.png" \
    --lda &
done

wait
echo "Scatter plots written under exp/graphs/${K}_components_CV=${CV_p}/scatter"
