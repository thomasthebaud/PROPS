#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"


# srun -p cpu python data/make_commonvoice_dataset.py CommonVoice &
# srun -p cpu python data/make_capspeech_gigaspeech_dataset.py GigaSpeech &
# srun -p cpu python data/make_mls_dataset.py MLS-en &
srun -p cpu python data/make_emilia_dataset.py Emilia-en

# wait 

# srun -p cpu python data/merge_csvs.py

# srun -p cpu python data/summary_table.py