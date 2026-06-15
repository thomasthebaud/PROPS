#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

N="1000"
input_dir="data/Capspeech_min100"
output_dir="data/Capspeech_min100_${N}"
rm -r "$output_dir"

srun -p cpu python bin/make_smaller_capspeech_set.py \
  --input-dir "$input_dir" \
  --output-dir "$output_dir" \
  --max-audios-per-profile "$N"

