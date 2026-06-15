#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"


# srun -p cpu python data/make_commonvoice_CS_dataset.py CommonVoiceCS
# srun -p cpu -c 16 python data/check_files_exists.py CommonVoiceCS --num-workers 16

# srun -p cpu -c 32 python data/make_commonvoice_dataset.py CommonVoice

# srun -p cpu -c 32 python data/make_gigaspeech_CS_dataset.py GigaSpeech --num-file-check-workers 32
# srun -p cpu -c 32 python data/check_files_exists.py GigaSpeech --num-workers 32

# srun -p cpu -c 16 python data/make_mls_dataset.py MLS-en --num-workers 16
# srun -p cpu -c 16 python data/check_files_exists.py MLS-en --num-workers 16

# srun -p cpu python data/make_emilia_dataset.py Emilia-en
# srun -p cpu -c 16 python data/check_files_exists.py Emilia-en --num-workers 16

# wait 

# echo "All datasets prepared. Now merging CSVs"
# srun -p cpu python data/merge_csvs.py

echo "Cleaning the labels to avoid duplicates (like high-pitch vs high-pitched)"
# srun -p cpu -c 32 python data/clean_duplicates.py --dataset Capspeech --num-workers 32 &
srun -p cpu -c 32 python data/clean_duplicates.py --dataset GigaSpeech --num-workers 32 &
srun -p cpu -c 32 python data/clean_duplicates.py --dataset CommonVoice --num-workers 32 &
srun -p cpu -c 32 python data/clean_duplicates.py --dataset MLS-en --num-workers 32 &
srun -p cpu -c 32 python data/clean_duplicates.py --dataset Emilia-en --num-workers 32 &

wait

echo "### SUMMARY TABLE ###"
srun -p cpu python data/summary_table.py
