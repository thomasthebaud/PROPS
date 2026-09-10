# ProPs: Prompted Profile Synthesis for Natural Language-Conditioned Speaker Embedding Generation

This repository contains the experiment pipeline for **ProPs**, a method for generating speaker-embedding distributions from natural-language speaker-profile descriptions. The current workflow builds a merged Capspeech-style metadata set, extracts ECAPA-TDNN x-vectors, creates profile descriptions, precomputes real profile GMMs, trains a composed GMM MDN, generates profile-conditioned GMMs, and evaluates whether generated samples preserve requested characteristics.
Those are the only files used to compute all results presented in the SLT 2026 article entitled the same way.

The top-level numbered shell scripts are the intended workflow entry points. They are written for a SLURM cluster and call Python jobs through `srun`.

## Current Pipeline

1. `00_prepare_metadata.sh`: prepare and clean dataset metadata, then write summary tables.
2. `01_extract_xvectors.sh`: extract ECAPA-TDNN x-vectors.
3. `02_get_descriptions.sh`: build profile combinations, merge them into Capspeech, generate descriptions, filter to `Capspeech_min100`, and build SBERT embeddings.
4. `02b_make_smaller_sets.sh`: optionally derive smaller Capspeech subsets.
5. `03_precompute_GMMs.sh`: fit one real-data GMM per profile for the main training/pretraining setup.
6. `04_pretrain_MDN.sh`: pretrain `ComposedGMM_MDN` routing from profile SBERT embeddings.
7. `06_fine_tune_MDN.sh`: fine-tune the pretrained composed MDN on per-sample Capspeech descriptions.
8. `07_inference.sh`: generate one GMM per test/dev row from a checkpoint.
9. `08_test_precomputed_GMMs.sh`: sweep precomputed real-data GMMs over multiple K values and evaluate Best-K behavior.
10. `09_compute_NLLs.sh`: generate MDN GMMs, score dev/test x-vectors, score baseline distributions, and build an NLL LaTeX table.
11. `10_characteristics_accuracy.sh`: train characteristic classifiers and evaluate generated GMM samples.
12. `11_scatter_plot.sh`: generate profile scatter/KDE figures and gender-only histogram overlays.
13. `12_confusion_matrices.sh`: generate log-likelihood confusion matrices.
14. `13_LDA_histograms.sh`: generate LDA/PCA histogram comparisons.

## Repository Layout

```text
.
|-- 00_prepare_metadata.sh
|-- 01_extract_xvectors.sh
|-- 02_get_descriptions.sh
|-- 02b_make_smaller_sets.sh
|-- 03_precompute_GMMs.sh
|-- 04_pretrain_MDN.sh
|-- 06_fine_tune_MDN.sh
|-- 07_inference.sh
|-- 08_test_precomputed_GMMs.sh
|-- 09_compute_NLLs.sh
|-- 10_characteristics_accuracy.sh
|-- 11_scatter_plot.sh
|-- 12_confusion_matrices.sh
|-- 13_LDA_histograms.sh
|-- bin/
|   |-- characteristics_accuracy.py
|   |-- compute_nll_scores.py
|   |-- compute_nll_scores_to_random.py
|   |-- create_GPT_descriptions.py
|   |-- extract_xvectors.py
|   |-- filter_capspeech_profiles.py
|   |-- find_valid_combinations.py
|   |-- finetune.py
|   |-- get_SBERT_embeddings.py
|   |-- inference.py
|   |-- make_nll_table.py
|   |-- merge_characteristics_accuracy.py
|   |-- plot_gender_histograms.py
|   |-- plot_profile_gmms.py
|   |-- precompute_GMMs.py
|   |-- pretrain.py
|   |-- profiles_summary.py
|   |-- test_precomputed_gmms.py
|   |-- train.py
|   |-- utils.py
|   `-- graphs/
|       |-- confusion_matrix.py
|       `-- lda_histograms.py
|-- data/
|   |-- clean_duplicates.py
|   |-- merge_csvs.py
|   |-- summary_table.py
|   |-- CommonVoice/
|   |-- Emilia-en/
|   |-- GigaSpeech/
|   |-- MLS-en/
|   |-- Capspeech/
|   `-- Capspeech_min100/
|-- exp -> /export/fs06/tthebau1/SHADOW/PROPS/exp
`-- environment.yml
```

`data/` stores metadata CSVs. `exp/` is a symlink to the experiment artifact directory and stores extracted x-vectors, SBERT embeddings, GMMs, checkpoints, metrics, and figures.

## Data Format

Dataset directories use split CSVs:

```text
data/<dataset>/train.csv
data/<dataset>/dev.csv
data/<dataset>/test.csv
```

Common columns are:

```text
dataset,id,duration,speaker,split,storage_path,sample_freq,pitch,age,gender,speaking_rate,speech_monotony,accent,capspeech_prompt
```

Capspeech-style profile tables include:

- `profile_combinations.csv`: valid profile combinations and `num_audios` counts.
- `unique_profile_combinations.csv`: one-known-field profiles used to ensure single-characteristic prompts exist.
- `profile_prompts.csv`: profile rows plus generated description columns such as `capspeech_desc` or `desc*`.
- `unique_profile_prompts.csv`: generated prompts for the single-known-field profiles.

The main filtered working dataset is currently `Capspeech_min100`, produced from `Capspeech` by keeping profiles with at least 100 utterances.

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate props
```

The scripts assume SLURM and CUDA for the GPU-heavy stages. On a non-SLURM machine, remove the `srun ...` prefix and set `--device cpu` or `--device cuda` as appropriate.

OpenAI credentials for prompt generation are loaded from `openai_keys.sh`; do not commit private keys.

## 0. Prepare and Clean Metadata

Run:

```bash
bash 00_prepare_metadata.sh
```

The dataset builders in `data/` create split metadata for `CommonVoice`, `GigaSpeech`, `MLS-en`, and `Emilia-en`. The active script then runs `data/clean_duplicates.py` on each dataset with 32 worker threads and finishes with `data/summary_table.py`.

`data/clean_duplicates.py` processes `train.csv`, `dev.csv`, and `test.csv` in parallel and writes the cleaned files in place. It reads large CSVs in chunks, prints one-line label sets for every profile field before and after cleaning, and reports how many dashed-accent rows were removed per split.

Current normalization rules are:

- `pitch`: `low-pitched` -> `low-pitch`; `medium-pitched` -> `moderate pitch`; `high-pitched` remains `high-pitched`.
- `speaking_rate`: `fast speed` -> `fast`; `measured speed` -> `moderate speed`; `slow speed` -> `slowly`.
- `accent`: rows whose non-unknown accent contains a dash, such as `british-american` or `british-guyanese`, are removed from the dataset.

## 1. Extract X-Vectors

Run:

```bash
bash 01_extract_xvectors.sh
```

This submits `bin/extract_xvectors.py` jobs. The extractor loads `speechbrain/spkrec-ecapa-voxceleb`, resamples audio to 16 kHz, computes one ECAPA embedding per utterance, and writes compressed `.npz` files under:

```text
exp/xvectors/ecapa_tdnn/<dataset>_<split>/xvectors/<id>.npz
```

It also writes extraction metadata such as `xvector.csv`, `failures.csv`, and `extract_args.json`.

## 2. Build Profiles, Descriptions, and SBERT Embeddings

Run all stages:

```bash
bash 02_get_descriptions.sh
```

Or run a single stage:

```bash
bash 02_get_descriptions.sh combinations
bash 02_get_descriptions.sh merging
bash 02_get_descriptions.sh unique
bash 02_get_descriptions.sh prompts
bash 02_get_descriptions.sh filter
bash 02_get_descriptions.sh sbert
bash 02_get_descriptions.sh summary
```

Stages:

- `combinations`: runs `bin/find_valid_combinations.py` per source dataset.
- `merging`: runs `bin/merge_profile_csvs.py` to merge source profile combinations into `data/Capspeech/`.
- `unique`: runs `bin/make_unique_profile_combinations.py` to build one-known-field profiles.
- `prompts`: runs `bin/create_GPT_descriptions.py` for profile prompts and prepends unique prompts to the main prompt table.
- `filter`: runs `bin/filter_capspeech_profiles.py` to create `data/Capspeech_min100/`.
- `sbert`: runs `bin/get_SBERT_embeddings.py` on `data/Capspeech_min100/profile_prompts.csv`.
- `summary`: runs `bin/profiles_summary.py`.

`bin/make_unique_profile_combinations.py` uses the profile fields `pitch`, `age`, `gender`, `speaking_rate`, `speech_monotony`, and `accent`.

## 3. Precompute Real Profile GMMs

Run:

```bash
bash 03_precompute_GMMs.sh
```

This fits one real-data GMM per Capspeech profile from train-set x-vectors. The script shards `bin/precompute_GMMs.py` jobs, writes profile `.npz` files under `profiles/`, and merges shard metadata into:

```text
exp/GMMs/<K>_components_precomputed/metadata.csv
exp/GMMs/<K>_components_precomputed/skipped_profiles.csv
```

Each profile GMM stores `pi_logits`, `pi`, `mu`, and diagonal `sigma`. Recent precompute/evaluation scripts use the cap rule `max_xvectors_per_profile=100*K` when sweeping K.

## 4. Pretrain the Composed MDN

Run:

```bash
bash 04_pretrain_MDN.sh
```

This runs `bin/pretrain.py` with `ComposedGMM_MDN`, `Capspeech_min100`, and profile SBERT embeddings from:

```text
exp/SBERT_embs/Capspeech_min100
```

The pretraining objective teaches the model's routing head to select the real profile-GMM component associated with each profile description. Checkpoints are written under `exp/MDN_models/` with names like:

```text
<K>_components_Capspeech_min100_p=1_pretrain.pt
```

## 5. Fine-Tune

Run:

```bash
bash 06_fine_tune_MDN.sh
```

This runs `bin/finetune.py`, loading a pretrained checkpoint and fine-tuning on per-utterance Capspeech descriptions from `data/Capspeech_min100/train.csv` and `dev.csv`. The current experiments use train fractions such as `0.1` and write checkpoints named like:

```text
exp/MDN_models/<K>_components_Capspeech_min100_p=0.1_finetune.pt
```

`bin/finetune.py` prints progress while preparing data: it announces CSV reads, reports how many rows were retained with non-empty descriptions, and logs x-vector matching progress every 100k scanned rows.

## 6. Generate Test/Dev GMMs

Run:

```bash
bash 07_inference.sh
```

This calls `bin/inference.py` with `--test-metadata-csv`, generating one GMM per row from a split CSV. The script chooses `capspeech_desc` when present and otherwise falls back to `capspeech_prompt`; `profile_prompts.csv` can provide fallback description text, but the saved GMM metadata fields still come from the split rows. Outputs are written under:

```text
exp/GMMs/<dataset>_<split>/<K>_components_<dataset>_p=<fraction>_<model>/
```

`bin/inference.py` now uses tqdm progress bars instead of repeated `Generated x/y files` prints, with descriptions such as `generating test-desc0`.

## 7. Evaluate Precomputed GMM K

Run all stages:

```bash
bash 08_test_precomputed_GMMs.sh
```

Or select stages:

```bash
bash 08_test_precomputed_GMMs.sh --precompute
bash 08_test_precomputed_GMMs.sh --evaluate
```

Stage 1 recomputes precomputed real-data GMMs into a separate root:

```text
exp/GMMs/stage08_precomputed/<K>_components_precomputed/
```

It loops sequentially over:

```text
K = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
```

For each K, it keeps profiles with `num_audios >= 2*K`, uses `max_xvectors_per_profile=100*K`, launches shard jobs, waits for that K to finish, merges shard metadata, and only then starts the next K.

Stage 2 runs `bin/test_precomputed_gmms.py` on dev/test x-vectors and writes Best-K tables, plots, and cache files under:

```text
exp/graphs/Best_K/
exp/Best_K/
```

## 8. NLL Scoring and Table

Run all stages:

```bash
bash 09_compute_NLLs.sh
```

Or select stages:

```bash
bash 09_compute_NLLs.sh --pretrain
bash 09_compute_NLLs.sh --finetune
bash 09_compute_NLLs.sh --scores
bash 09_compute_NLLs.sh --random
bash 09_compute_NLLs.sh --table
```

Stages:

- Stage 1 generates pretrained MDN GMMs for test `desc0`, test `capspeech_prompt`, dev `desc1`, and dev `capspeech_prompt`.
- Stage 2 generates the same GMM sets with the finetuned MDN.
- Stage 3 launches separate `bin/compute_nll_scores.py` jobs for pretrained, pretrained-with-uniform-pi, and finetuned GMMs. For `desc0`/`desc1`, each x-vector is scored against the GMM matching its profile. For `capspeech_prompt`, each x-vector is scored against the GMM generated from that specific prompt row.
- Stage 4 runs `bin/compute_nll_scores_to_random.py`, scoring real dev/test x-vectors against `N(0,1)` and random K-component GMM baselines.
- Stage 5 runs `bin/make_nll_table.py`, which tolerates missing CSVs by warning and rendering `NA`, writes `exp/NLL_scores/table.tex`, and copies it to `exp/tables/nll_scores.tex`.

Score CSVs are written under:

```text
exp/NLL_scores/jobs/
exp/NLL_scores/random/
```

`bin/compute_nll_scores.py`, `bin/compute_nll_scores_to_random.py`, and `bin/make_nll_table.py` include tqdm progress bars and status prints for loading, scoring, and table aggregation.

## 9. Characteristic Accuracy Evaluation

Run:

```bash
bash 10_characteristics_accuracy.sh
```

This runs one CPU job per characteristic using `bin/characteristics_accuracy.py`, then merges the per-characteristic CSVs with `bin/merge_characteristics_accuracy.py`. The current script evaluates:

```text
gender, age, accent, speech_monotony, pitch, speaking_rate
```

Important behavior:

- The classifier is trained on a configurable real x-vector dataset, currently `Capspeech_min100_dev`, and tested on generated samples for `Capspeech_min100_test` GMM metadata.
- Labels with fewer than `--min-label-xvectors` real x-vectors are skipped before classifier training and generated-label evaluation.
- Ordinal characteristics can use `--ordinal-classifier-mode`, currently `svr` in the script.
- Age, pitch, speaking-rate, and speech-monotony labels are ordered semantically in confusion matrices rather than alphabetically.
- Pitch and speaking-rate aliases are normalized through `bin/utils.py`.
- The merged outputs are `accuracies.csv` and `results.tex`.

Outputs are written under:

```text
exp/characteristics_prediction/<run_name>_<min_labels>/
exp/characteristics_prediction/<run_name>_<min_labels>/per_characteristic/
```

## 10. Scatter, KDE, and Histogram Plots

Run:

```bash
bash 11_scatter_plot.sh
```

`11_scatter_plot.sh` uses `bin/plot_profile_gmms.py` for multi-profile scatter/KDE plots. For the active accent comparison, it samples generated GMMs, gathers matching real x-vectors, projects them with LDA when `--lda` is passed, and writes:

```text
exp/graphs/<run_name>/scatter/K=<K>_<profiles>_test.png
exp/graphs/<run_name>/scatter/K=<K>_<profiles>_test_all_layers.png
```

The main plot contains four subplots: test-set KDE, generated KDE, all layers, and test-set x-vectors only. The `_all_layers` companion contains only the all-layers subplot and restores the layer legend. Scatter axis labels use `LDA - dimension 1` and `LDA - dimension 2` for LDA plots.

`bin/plot_gender_histograms.py` is a gender-only helper used for `gender=male,female` with all other requested fields unknown. It treats unknown metadata fields as wildcards when matching generated GMMs, samples a total of `--samples` generated x-vectors from the found GMMs for each gender, and overlays real vs generated histograms on one axis. Real histograms use dotted lines; generated histograms use solid lines; male is plotted in two blues and female in two oranges.

## 11. Log-Likelihood Confusion Matrices

Run:

```bash
bash 12_confusion_matrices.sh
```

This calls `bin/graphs/confusion_matrix.py`. It computes the mean log-likelihood of real x-vectors from each label under generated GMMs for each generated label. The current script runs `age`, `gender`, and `accent` matrices. The age matrix excludes `child`, and label ordering uses the shared semantic ordering from `bin/utils.py`.

Outputs include:

```text
<category>_confusion_matrix.csv
<category>_confusion_counts.csv
<category>_confusion_matrix.png
```

## 12. LDA/PCA Histograms

Run:

```bash
bash 13_LDA_histograms.sh
```

This calls `bin/graphs/lda_histograms.py`. The current active configuration writes PCA histograms for accent-only and gender-only conditions under:

```text
exp/graphs/<run_name>/
```

The script keeps LDA histogram calls commented for quick switching when needed.

## Shared Label Ordering and Normalization

`bin/utils.py` centralizes label ordering and alias handling used by classifier evaluation and graph scripts.

Current semantic orders:

- Age: `child`, `teenager`, `young adult`, `middle-aged adult`, `elderly`, then decade labels from `teens` through `nineties`.
- Pitch: `low-pitch`, `moderate pitch`, `high-pitch`.
- Speaking rate: `slow`, `slightly slowly`, `moderate speed`, `slightly fast`, `fast`.
- Speech monotony: `monotone`, `slightly expressive and animated`, `expressive and animated`, `very expressive and animated`.

Unknown labels fall back to sorted order after known labels.

## Artifact Conventions

Common artifact roots:

- `exp/xvectors/ecapa_tdnn/`: extracted SpeechBrain ECAPA x-vectors.
- `exp/SBERT_embs/`: SBERT embeddings for generated profile descriptions.
- `exp/GMMs/<name>/`: precomputed or generated profile GMMs.
- `exp/MDN_models/`: trained PyTorch checkpoints.
- `exp/characteristics_prediction/`: SVM-based generated-sample accuracy outputs.
- `exp/graphs/`: figures and plotting CSVs.
- `exp/NLL_scores/`: NLL score jobs, random-baseline scores, summary CSVs, and `table.tex`.
- `exp/tables/`: copied LaTeX tables for paper/report use.

GMM `.npz` files use:

- `pi_logits`: log mixture weights.
- `pi`: normalized mixture weights.
- `mu`: component means, shape `[K, xvector_dim]`.
- `sigma`: diagonal component standard deviations, shape `[K, xvector_dim]`.

Model checkpoints contain `model_state_dict`, optimizer state when applicable, loss metadata, `model_config`, dataset settings, and regularization settings.

## Running Without SLURM

The scripts use commands like:

```bash
srun -p gpu --gpus 1 python ...
srun -p cpu python ...
```

On a non-SLURM machine, remove the `srun ...` prefix and run the Python commands directly.

## Citation
Arxiv link will be replaced once the SLT 2026 publication is out.
```bibtex
@article{thebaud2026props,
  title={ProPS: Prompted Profile Synthesis for Natural Language-Conditioned Speaker Embedding Distributions},
  author={Thebaud, Thomas and Lee, Junhyeok and Moro-Velazquez, Laureano and Lopez, Jesus Villalba and Dehak, Najim},
  journal={arXiv preprint arXiv:2607.05276},
  year={2026}
}

```
