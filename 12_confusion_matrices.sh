#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=4
CV_p=1
N_min=10

mkdir -p "exp/graphs/${K}_components_CV=${CV_p}"

for category in age gender accent; do
  echo "Computing ${category} log-likelihood confusion matrix"
  srun -p cpu python bin/graphs/confusion_matrix.py \
    --category "$category" \
    --test-csv-paths "CommonVoice_test" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gmm-metadata-csv "exp/GMMs/${K}_components_CV=${CV_p}_mixed/metadata.csv" \
    --output-dir "exp/graphs/${K}_components_CV=${CV_p}" \
    --n-min "$N_min"
done

echo "Confusion matrices written under exp/graphs/${K}_components_CV=${CV_p}"
