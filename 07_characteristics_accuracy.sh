#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=4
CV_p=1
N=1000

echo "Evaluating characteristic SVM accuracies"
mkdir -p "exp/characteristics_prediction/${K}_components_CV=${CV_p}"

srun -p cpu python bin/characteristics_accuracy.py \
  --test-csv-paths "CommonVoice_test" \
  --xvector-root exp/xvectors/ecapa_tdnn \
  --gmm-metadata-csv "exp/GMMs/${K}_components_CV=${CV_p}_mixed/metadata.csv" \
  --output-csv "exp/characteristics_prediction/${K}_components_CV=${CV_p}/accuracies.csv" \
  --samples "$N"
