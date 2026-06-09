#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=4
CV_p=1
N=5000
N_min=50

output_dir="exp/graphs/${K}_components_CV=${CV_p}"
gmm_metadata_csv="exp/GMMs/${K}_components_CV=${CV_p}_mixed/metadata.csv"
mkdir -p "$output_dir"

accents=(
  "Africa"
  "Australia"
  "Canada"
  "Caribbean"
  "East Asia"
  "England"
  "French"
  "Germanic"
  "Ireland"
  "Latin America"
  "Middle East"
  "New Zealand"
  "Nordic"
  "Scotland"
  "Slavic / Eastern Europe"
  "South Asia"
  "Southeast Asia"
  "Southern Europe"
  "US"
  "Wales"
)

echo "Plotting accent-only PCA histograms"
for accent in "${accents[@]}"; do
  srun -p cpu python bin/graphs/lda_histograms.py \
    --test-csv-paths "CommonVoice_test" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gmm-metadata-csv "$gmm_metadata_csv" \
    --output-dir "$output_dir" \
    --pitch "unknown" \
    --age "unknown" \
    --gender "unknown" \
    --speaking-rate "unknown" \
    --speech-monotony "unknown" \
    --accent "$accent" \
    --samples "$N" \
    --n-min "$N_min" --pca &
done

echo "Plotting accent-only LDA histograms"
for accent in "${accents[@]}"; do
  srun -p cpu python bin/graphs/lda_histograms.py \
    --test-csv-paths "CommonVoice_test" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gmm-metadata-csv "$gmm_metadata_csv" \
    --output-dir "$output_dir" \
    --pitch "unknown" \
    --age "unknown" \
    --gender "unknown" \
    --speaking-rate "unknown" \
    --speech-monotony "unknown" \
    --accent "$accent" \
    --samples "$N" \
    --n-min "$N_min" &
done

wait

genders=(
  "male"
  "female"
)

echo "Plotting gender-only LDA histograms"
for gender in "${genders[@]}"; do
  srun -p cpu python bin/graphs/lda_histograms.py \
    --test-csv-paths "CommonVoice_test" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gmm-metadata-csv "$gmm_metadata_csv" \
    --output-dir "$output_dir" \
    --pitch "unknown" \
    --age "unknown" \
    --gender "$gender" \
    --speaking-rate "unknown" \
    --speech-monotony "unknown" \
    --accent "unknown" \
    --samples "$N" \
    --n-min "$N_min" &
done

echo "Plotting gender-only PCA histograms"
for gender in "${genders[@]}"; do
  srun -p cpu python bin/graphs/lda_histograms.py \
    --test-csv-paths "CommonVoice_test" \
    --xvector-root exp/xvectors/ecapa_tdnn \
    --gmm-metadata-csv "$gmm_metadata_csv" \
    --output-dir "$output_dir" \
    --pitch "unknown" \
    --age "unknown" \
    --gender "$gender" \
    --speaking-rate "unknown" \
    --speech-monotony "unknown" \
    --accent "unknown" \
    --samples "$N" \
    --n-min "$N_min" --pca &
done

wait

echo "LDA histograms written under ${output_dir}"
