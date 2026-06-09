

# Generate a csv with all possible valid combinations (4970 valid combinations if using Gigaspeech & CommonVoice)
# here only CommonVoice to start
# srun -p cpu python bin/find_valid_combinations.py

# Get GPT prompts for each combination (10 per combination: 49,700 prompts)
# when using only CommonVoice, 4,050 prompts

source openai_keys.sh

srun -p cpu python bin/create_GPT_descriptions.py \
    --input data/profile_combinations.csv \
    --output data/profile_prompts.csv \
    --model "gpt-5.4-mini-2026-03-17" \
    --org $org_key \
    --key $openai_api_key

# Generate SBERT embeddings of descriptions
srun -p gpu --gpus 1 python bin/get_SBERT_embeddings.py \
    --input data/profile_prompts.csv \
    --output-dir exp/SBERT_embs/ \
    --model "sentence-transformers/all-MiniLM-L6-v2" \
    --device "cuda:0" 