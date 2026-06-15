#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

K=1
dataset_name="Capspeech_min100"
model='pretrain'
train_fraction=1
run_name="${K}_components_${dataset_name}_p=${train_fraction}_${model}"
N=5000
N_min=50

output_dir="exp/graphs/${run_name}"
gmm_metadata_csv="exp/GMMs/${run_name}_desc0/metadata.csv"
mkdir -p "$output_dir"

accents=(
  "american"
  "australian"
  "belgian"
  "brazilian"
  "british"
  "british-american"
  "british-guyanese"
  "canadian"
  "cantonese"
  "colombian-american"
  "czech"
  "dutch"
  "english"
  "filipino"
  "french"
  "german"
  "hungarian"
  "indian"
  "indian-american"
  "irish"
  "italian"
  "japanese"
  "mexican"
  "new zealand"
  "norwegian"
  "portuguese"
  "russian"
  "scottish"
  "slovenian"
  "welsh"
)

echo "Plotting accent-only PCA histograms"
for accent in "${accents[@]}"; do
  srun -p cpu python bin/graphs/lda_histograms.py \
    --test-csv-paths "$dataset_name" \
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

# echo "Plotting accent-only LDA histograms"
# for accent in "${accents[@]}"; do
#   srun -p cpu python bin/graphs/lda_histograms.py \
#     --test-csv-paths "$dataset_name" \
#     --xvector-root exp/xvectors/ecapa_tdnn \
#     --gmm-metadata-csv "$gmm_metadata_csv" \
#     --output-dir "$output_dir" \
#     --pitch "unknown" \
#     --age "unknown" \
#     --gender "unknown" \
#     --speaking-rate "unknown" \
#     --speech-monotony "unknown" \
#     --accent "$accent" \
#     --samples "$N" \
#     --n-min "$N_min" &
# done

wait

genders=(
  "male"
  "female"
)

# echo "Plotting gender-only LDA histograms"
# for gender in "${genders[@]}"; do
#   srun -p cpu python bin/graphs/lda_histograms.py \
#     --test-csv-paths "$dataset_name" \
#     --xvector-root exp/xvectors/ecapa_tdnn \
#     --gmm-metadata-csv "$gmm_metadata_csv" \
#     --output-dir "$output_dir" \
#     --pitch "unknown" \
#     --age "unknown" \
#     --gender "$gender" \
#     --speaking-rate "unknown" \
#     --speech-monotony "unknown" \
#     --accent "unknown" \
#     --samples "$N" \
#     --n-min "$N_min" &
# done

echo "Plotting gender-only PCA histograms"
for gender in "${genders[@]}"; do
  srun -p cpu python bin/graphs/lda_histograms.py \
    --test-csv-paths "$dataset_name" \
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
