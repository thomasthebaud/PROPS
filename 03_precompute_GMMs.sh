#!/usr/bin/env bash
set -euo pipefail

K=4

echo "Precomputing one K=${K} GMM per fully-known profile from all of CommonVoice_train"

srun -p cpu python bin/precompute_GMMs.py \
  --train-csv-paths "CommonVoice_train" \
  --data-root data \
  --xvector-root exp/xvectors/ecapa_tdnn \
  --profile-combinations-csv data/profile_combinations.csv \
  --output-dir exp/GMMs/${K}_components_precomputed \
  --K ${K}
