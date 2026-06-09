# ProPs: Prompted Profile Synthesis for Natural Language-Conditioned Speaker Embedding Generation

This repository contains the experiment pipeline for **ProPs**, a method for generating speaker-embedding distributions from natural language descriptions of speaker profiles. The code builds profile descriptions such as "a thirties male speaker with a US accent", embeds those descriptions with SBERT, and trains a mixture density network (MDN) to predict Gaussian mixture models (GMMs) in ECAPA-TDNN speaker-embedding space.

At a high level, the pipeline does the following:

1. Builds metadata tables for CommonVoice and CapSpeech/GigaSpeech-style data.
2. Extracts ECAPA-TDNN x-vectors for each utterance.
3. Enumerates valid speaker-profile combinations.
4. Generates multiple natural language prompts for each profile.
5. Converts prompts to SBERT embeddings.
6. Fits real-data profile GMMs from x-vectors.
7. Trains a text-conditioned MDN.
8. Uses the MDN to synthesize one GMM per prompt/profile.
9. Evaluates whether the generated GMMs preserve requested profile characteristics.
10. Produces scatter plots, confusion matrices, and LDA/PCA histograms.

The top-level `01_*.sh` through `13_*.sh` scripts are the intended workflow entry points. They are written for a SLURM cluster and call Python jobs through `srun`.

## Repository Layout

```text
.
|-- 01_extract_xvectors.sh
|-- 02_get_descriptions.sh
|-- 03_precompute_GMMs.sh
|-- 04_train_MDN.sh
|-- 05_fine_tune_MDN.sh
|-- 06_inference.sh
|-- 07_characteristics_accuracy.sh
|-- 11_scatter_plot.sh
|-- 12_confusion_matrices.sh
|-- 13_LDA_histograms.sh
|-- bin/
|   |-- characteristics_accuracy.py
|   |-- create_GPT_descriptions.py
|   |-- extract_xvectors.py
|   |-- find_valid_combinations.py
|   |-- get_SBERT_embeddings.py
|   |-- inference.py
|   |-- model.py
|   |-- my_dataset.py
|   |-- plot_profile_gmms.py
|   |-- precompute_GMMs.py
|   |-- test_metrics.py
|   |-- train.py
|   |-- utils.py
|   `-- graphs/
|       |-- confusion_matrix.py
|       `-- lda_histograms.py
|-- data/
|   |-- CommonVoice_dev/
|   |-- CommonVoice_test/
|   |-- CommonVoice_train/
|   |-- GigaSpeech_test/
|   |-- GigaSpeech_train/
|   |-- make_capspeech_gigaspeech_dataset.py
|   |-- make_commonvoice_dataset.py
|   |-- profile_combinations.csv
|   `-- profile_prompts.csv
|-- exp -> /export/fs06/tthebau1/SHADOW/PROPS/exp
`-- visuals.ipynb
```

`data/` holds dataset metadata CSVs. `exp/` is a symlink to the experiment artifact directory and stores extracted x-vectors, SBERT embeddings, precomputed and generated GMMs, trained MDN checkpoints, metrics, and figures.

## Data Format

Each dataset directory under `data/` is expected to contain:

- `segments.csv`: one row per utterance segment. Required column: `id`. Common columns include `speaker`, `gender`, `age`, `accent`, `pitch`, `speaking_rate`, and `speech_monotony`.
- `recordings.csv`: maps each segment `id` to a `storage_path`, `duration`, and `sample_freq`.

CommonVoice data currently contains `id`, `gender`, `age`, `speaker`, `transcript`, and `accent`. GigaSpeech/CapSpeech metadata may additionally contain prosodic fields such as `pitch`, `speaking_rate`, and `speech_monotony`.

The profile tables are:

- `data/profile_combinations.csv`: valid profile combinations represented by the available data. In the current checked-in table, the active fields are `gender`, `age`, and `accent`.
- `data/profile_prompts.csv`: profile fields plus `desc0` through `desc9`. Each `desc*` column is a natural language description of the same profile.

By convention, `desc1` through `desc9` are used by the training dataset and `desc0` is reserved for inference/evaluation.

## Environment

The conda environment for this repository is pinned in `environment.yml`. Create it with:

```bash
conda env create -f environment.yml
conda activate props
```

The environment includes the Python, CUDA/PyTorch, audio, embedding, OpenAI, plotting, and scikit-learn dependencies used by the pipeline. The top-level scripts still assume SLURM is available for `srun`; on a non-SLURM machine, remove the `srun` prefix as described below.

The scripts assume CUDA is available for x-vector extraction, SBERT embedding, and MDN training unless the command-line `--device` options are changed.

Before publishing or running prompt generation, configure OpenAI credentials securely for your environment. Do not commit private API keys.

## Full Workflow

### 0. Build Dataset Metadata

The dataset preparation scripts live in `data/`.

`data/make_commonvoice_dataset.py` reads CommonVoice TSVs from `/export/corpora7/CommonVoice-19.0/en/`, filters speakers with known age, gender, duration, and usable accent labels, maps detailed accent strings into broader accent categories, and writes `segments.csv` and `recordings.csv` for the CommonVoice train, test, and dev splits.

`data/make_capspeech_gigaspeech_dataset.py` reads CapSpeech/GigaSpeech metadata, merges prompt/caption metadata with audio metadata, removes missing audio files, and writes `segments.csv` and `recordings.csv` for GigaSpeech splits.

These scripts contain local filesystem paths and may need editing for a new machine.

### 1. Extract Speaker X-Vectors

Run:

```bash
bash 01_extract_xvectors.sh
```

This submits `bin/extract_xvectors.py` for `CommonVoice_dev`, `CommonVoice_test`, `GigaSpeech_test`, `CommonVoice_train`, and `GigaSpeech_train`.

The extractor loads `speechbrain/spkrec-ecapa-voxceleb`, resamples audio to 16 kHz, computes one ECAPA embedding per segment, and writes compressed `.npz` files under:

```text
exp/xvectors/ecapa_tdnn/<dataset>/xvectors/<id>.npz
```

It also writes `xvector.csv`, `failures.csv`, and `extract_args.json`. The downstream training code normalizes x-vectors to unit length and treats them as 192-dimensional targets.

### 2. Generate Profile Descriptions and SBERT Embeddings

Run:

```bash
bash 02_get_descriptions.sh
```

This script has two jobs. First, `bin/create_GPT_descriptions.py` reads `data/profile_combinations.csv` and writes `data/profile_prompts.csv`. For every valid profile, it asks the OpenAI model for 10 concise natural language descriptions and stores them as `desc0` through `desc9`.

Second, `bin/get_SBERT_embeddings.py` embeds every description with `sentence-transformers/all-MiniLM-L6-v2` and writes:

```text
exp/SBERT_embs/<profile_index>/desc0.npz
exp/SBERT_embs/<profile_index>/desc1.npz
...
exp/SBERT_embs/<profile_index>/desc9.npz
```

Each file stores the SBERT vector, row index, description column, original text, and model name. The MDN uses 384-dimensional SBERT embeddings as inputs.

If you need to rebuild `profile_combinations.csv`, use `bin/find_valid_combinations.py`. It enumerates profile-field combinations, keeps combinations represented in the metadata, removes the all-unknown combination, and writes `data/profile_combinations.csv`.

### 3. Precompute Real Profile GMMs

Run:

```bash
bash 03_precompute_GMMs.sh
```

This calls `bin/precompute_GMMs.py` with `K=4`, using all of `CommonVoice_train`. For each fully known profile in `data/profile_combinations.csv`, it finds matching real x-vectors and fits a diagonal-covariance GMM.

Outputs are written under:

```text
exp/GMMs/4_components_precomputed/
```

Important files include `profiles/*.npz`, `metadata.csv`, and `skipped_profiles.csv`. Each GMM file contains `pi_logits`, `pi`, `mu`, and `sigma`.

### 4. Train the MDN Baseline

Run:

```bash
bash 04_train_MDN.sh
```

This trains `bin/train.py` with `model-name GaussianMDN`, `K=128`, and a 5 percent subset of `CommonVoice_train`. The model maps SBERT prompt embeddings to an x-vector GMM:

```text
SBERT description embedding -> MDN -> pi, mu, sigma over x-vector space
```

The default script uses 100 epochs, batch size 256, hidden layers `1024,2048,2048,1024`, entropy regularization, pi regularization, and AdamW optimization. The best checkpoint by dev negative log-likelihood is saved to:

```text
exp/MDN_models/128_components_CV=0.05_GS=0.pt
```

### 5. Train or Fine-Tune the Composed GMM MDN

Run:

```bash
bash 05_fine_tune_MDN.sh
```

This trains `bin/train.py` with `model-name ComposedGMM_MDN`, `K=4`, and all of `CommonVoice_train`.

`ComposedGMM_MDN` loads all precomputed profile GMM components from:

```text
exp/GMMs/4_components_precomputed/
```

It concatenates those components into one large component bank. The network then learns text-conditioned mixture weights over that bank while the loaded means and variances are frozen by default. This is the central composition mechanism: natural language selects and mixes real profile components in speaker-embedding space.

The checkpoint is saved to:

```text
exp/MDN_models/4_components_CV=1.pt
```

### 6. Generate Profile-Conditioned GMMs

Run:

```bash
bash 06_inference.sh
```

This calls `bin/inference.py`, loads the trained composed MDN checkpoint, embeds each profile through the reserved `desc0` column, and writes one generated GMM per matching profile.

Outputs are written under:

```text
exp/GMMs/4_components_CV=1_mixed/
```

Important files are `profiles/profile_<index>_desc0_gmm.npz` and `metadata.csv`. During inference, low-weight components are pruned and the remaining weights are renormalized.

### 7. Characteristic Accuracy Evaluation

Run:

```bash
bash 07_characteristics_accuracy.sh
```

This calls `bin/characteristics_accuracy.py`. For each characteristic, usually `gender`, `age`, and `accent`, it loads real test-set x-vectors, trains a balanced RBF SVM classifier on known labels, samples generated x-vectors from the corresponding generated GMM, and measures whether the classifier predicts the requested label.

Outputs are written under:

```text
exp/characteristics_prediction/4_components_CV=1/
```

Important files include `accuracies.csv` and `confusion_<characteristic>.png`.

### 8. Scatter Plots

Run:

```bash
bash 11_scatter_plot.sh
```

This calls `bin/plot_profile_gmms.py` for selected age/gender and accent/gender comparisons. It samples from generated GMMs, fits test-set ground-truth GMMs for matching real x-vectors, and projects both distributions with PCA and optionally LDA.

Outputs are written under:

```text
exp/graphs/4_components_CV=1/scatter/
```

The plots show generated KDEs, test-set KDEs, sampled x-vectors, generated centers, and ground-truth centers.

### 9. Log-Likelihood Confusion Matrices

Run:

```bash
bash 12_confusion_matrices.sh
```

This calls `bin/graphs/confusion_matrix.py` for `age`, `gender`, and `accent`. It computes the mean log-likelihood of real x-vectors from each label under generated GMMs for each label.

Outputs include:

```text
exp/graphs/4_components_CV=1/<category>_confusion_matrix.csv
exp/graphs/4_components_CV=1/<category>_confusion_counts.csv
exp/graphs/4_components_CV=1/<category>_confusion_matrix.png
```

Rows are real labels and columns are generated profile GMM labels.

### 10. LDA and PCA Histograms

Run:

```bash
bash 13_LDA_histograms.sh
```

This calls `bin/graphs/lda_histograms.py`. For each requested profile, it samples generated x-vectors, gathers matching real x-vectors, projects real and generated samples onto one dimension with LDA or PCA, and plots density histograms/KDEs.

The script currently loops over accent-only and gender-only profiles, writing figures under:

```text
exp/graphs/4_components_CV=1/
```

## Additional Modules

- `bin/model.py`: defines `GaussianMDN`, `LearnablePi_MDN`, and `ComposedGMM_MDN`.
- `bin/my_dataset.py`: defines PyTorch datasets that match profile prompt embeddings to real x-vectors. `ProfileDataset` samples utterances matching a profile and uses `desc1` through `desc9` for training rows.
- `bin/test_metrics.py`: computes test-set x-vector log-likelihoods under generated GMMs and writes summary CSVs plus full likelihood lists.
- `bin/utils.py`: shared GMM, x-vector loading, sampling, normalization, CSV, and condition-matching utilities.
- `visuals.ipynb`: minimal scratch notebook for plotting/data inspection.

## Model Summary

`GaussianMDN` is the direct baseline. It predicts all GMM parameters from the text embedding:

```text
embedding -> pi_logits, mu, sigma
```

`LearnablePi_MDN` keeps shared learnable means and variances and predicts mixture weights from text.

`ComposedGMM_MDN` loads precomputed profile GMM means and variances from disk and predicts mixture weights over those existing components. This lets generated speaker distributions be assembled from real x-vector profile components while remaining controllable from natural language.

Training minimizes negative log-likelihood of real x-vectors under the predicted GMM. Optional losses include entropy regularization, mean-norm regularization, variance reduction, and pi regularization.

## Artifact Conventions

Common artifact roots:

- `exp/xvectors/ecapa_tdnn/`: extracted SpeechBrain ECAPA x-vectors
- `exp/SBERT_embs/`: SBERT embeddings for `desc0` through `desc9`
- `exp/GMMs/<name>/`: precomputed or generated profile GMMs
- `exp/MDN_models/`: trained PyTorch checkpoints
- `exp/characteristics_prediction/`: SVM-based generated-sample accuracy outputs
- `exp/graphs/`: figures and plotting CSVs

GMM `.npz` files use:

- `pi_logits`: log mixture weights
- `pi`: normalized mixture weights
- `mu`: component means, shape `[K, xvector_dim]`
- `sigma`: diagonal component standard deviations, shape `[K, xvector_dim]`

Model checkpoints contain `model_state_dict`, `optimizer_state_dict`, loss metadata, `model_config`, dataset settings, and regularization settings.

## Running Without SLURM

The shell scripts use commands like:

```bash
srun -p gpu --gpus 1 python ...
srun -p cpu python ...
```

On a non-SLURM machine, remove the `srun ...` prefix and run the Python commands directly, adjusting `--device cpu` or `--device cuda` as appropriate.

## Citation

```bibtex
@misc{
  title         = {},
  archivePrefix = {arXiv},
  eprint        = {},
  year          = {},
}
```
