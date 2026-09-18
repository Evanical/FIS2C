# FIS2C-SteerMusic

This repository contains the implementation of **FIS2C**, built on top of [SteerMusic](https://arxiv.org/abs/2504.10826) for zero-shot text-guided and personalized music editing.

## Environment Setup

FIS2C follows the environment of SteerMusic. The original SteerMusic implementation was tested with:

- Python 3.8.10
- PyTorch 2.2.0 + CUDA 12.1
- AudioLDM2

We recommend first creating the SteerMusic environment.

```bash
conda create -n fis2c python=3.8.10
conda activate fis2c
```

Install PyTorch:

```bash
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the SteerMusic dependencies:

```bash
pip install -r requirements.txt
```

The FIS2C training and batch scripts additionally use `pandas` and `pyarrow` for metadata loading:

```bash
pip install pandas pyarrow
```

If the following Hugging Face compatibility error occurs:

```text
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

follow the original SteerMusic setup and replace `cached_download` with `hf_hub_download` in:

```text
diffusers/utils/dynamic_modules_utils.py
```

### Pretrained models

FIS2C relies on the same pretrained AudioLDM2 backbone as SteerMusic. Make sure the AudioLDM2 checkpoint can be loaded by the SteerMusic code.

For feedback-based inference and evaluation, the current code also uses:

- CLAP checkpoint
- MERT/deTAC model

The default CLAP checkpoint arguments in the scripts are:

```text
clap/pretrained/music_audioset_epoch_15_esc_90.14.pt
```

and the default deTAC backbone is:

```text
m-a-p/MERT-v1-95M
```

For offline inference, a local AudioLDM2 directory can be specified with:

```bash
--audioldm2_path /path/to/AudioLDM2
```

---

## Dataset Preparation

### ZoME-Bench

The current FIS2C training script uses **ZoME-Bench** metadata.

The metadata should contain at least:

```text
original_prompt
editing_prompt
audio_path
```

The training script accepts:

```text
.parquet
.csv
.json
.jsonl
.ndjson
```

ZoME-Bench metadata may provide YouTube identifiers and segment timestamps rather than packaged audio files. Download and crop the corresponding audio clips first, then store them locally and pass the audio directory through `--audio_root`.

A typical data directory is:

```text
ZoME-Bench/
├── metadata_with_audio.parquet
└── audio/
    ├── xxx_0_10.wav
    ├── xxx_10_20.wav
    └── ...
```

If `audio_path` is relative, FIS2C resolves it using `--audio_root`. When `ytid`, `start_s`, and `end_s` are available, the loader also searches for files named as:

```text
{ytid}_{start_s}_{end_s}.wav
```

---

## Training FIS2C

Train the FIS2C controller with:

```bash
python train_fis2c.py \
    --metadata /path/to/ZoME-Bench/metadata_with_audio.parquet \
    --audio_root /path/to/ZoME-Bench/audio \
    --output_dir ./fisc_checkpoints \
    --device cuda:0 \
    --epochs 3 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --guidance_scale 30 \
    --lambda_max 0.03 \
    --fisc_audio_feature_dim 32000 \
    --beta_mag 0.20 \
    --beta_delta 0.05 \
    --beta_lambda 0.01 \
    --beta_over 0.20 \
    --max_grad_norm 1.0 \
    --semantic_reward_scale 100 \
    --structural_reward_scale 1.0 \
    --save_every 500
```

During training, the semantic and structural feedback terms are implemented as differentiable score-level surrogates at sampled DDS states. CLAP and deTAC are used for feedback-based inference and evaluation rather than being directly back-propagated through during controller training.

The final checkpoint is saved as:

```text
./fisc_checkpoints/fisc_final.pt
```

For a quick test:

```bash
python train_fis2c.py \
    --metadata /path/to/ZoME-Bench/metadata_with_audio.parquet \
    --audio_root /path/to/ZoME-Bench/audio \
    --output_dir ./fisc_checkpoints_debug \
    --device cuda:0 \
    --epochs 1 \
    --max_samples 2 \
    --save_every 1
```

---

## Zero-shot Text-guided Music Editing

Run a single example with:

```bash
python fis2c_edit.py \
    --audio_path /path/to/source.wav \
    --prompt_ref "Energetic piano cover with a groovy, reverberant melody." \
    --prompt "Energetic guitar cover with a groovy, reverberant melody." \
    --output_dir ./outputs/example \
    --fisc_ckpt ./fisc_checkpoints/fisc_final.pt \
    --device cuda:0 \
    --validation_step 500 \
    --guidance_scale 30 \
    --lambda_max 0.03 \
    --fisc_audio_feature_dim 32000 \
    --fisc_strength 1.3 \
    --edit_strength 1.3 \
    --feedback_mode alternate_clap_detac \
    --feedback_switch_every 25
```

Available feedback modes are:

```text
only_clap
only_detac
clap_detac
no_feedback
alternate_clap_detac
clap_then_detac
detac_then_clap
cycle_clap_detac_both
```

---

## Batch Zero-shot Editing

For batch editing:

```bash
python run_batch_fis2c_edit.py \
    --metadata /path/to/metadata.parquet \
    --audio_root /path/to/audio \
    --output_root ./outputs/zero_shot \
    --fisc_ckpt ./fisc_checkpoints/fisc_final.pt \
    --device cuda:0 \
    --validation_step 500 \
    --guidance_scale 30 \
    --lambda_max 0.03 \
    --fisc_audio_feature_dim 32000 \
    --fisc_strength 1.3 \
    --edit_strength 1.3 \
    --feedback_mode alternate_clap_detac \
    --feedback_switch_every 25
```

The batch script writes:

```text
batch_status.jsonl
```

under the output directory and stores generated audio in per-sample folders.

---

## Personalized Music Editing

FIS2C personalized editing is built on **SteerMusic+**. As in the original SteerMusic implementation, a personalized AudioLDM2 checkpoint is required. The personalized diffusion model can be obtained following the DreamSound fine-tuning procedure used by SteerMusic+.

The source prompt must contain the edited concept inside square brackets, for example:

```text
Energetic [piano] cover with a groovy, reverberant melody.
```

Run personalized editing with:

```bash
python fis2c_personalized_edit.py \
    --audio_path /path/to/source.wav \
    --prompt_ref "Energetic [piano] cover with a groovy, reverberant melody." \
    --concept bouzouki \
    --personalized_ckpt /path/to/personalized/checkpoint \
    --fisc_ckpt ./fisc_checkpoints/fisc_final.pt \
    --output_dir ./outputs/personalized_example \
    --device cuda:0 \
    --validation_step 500 \
    --guidance_scale 30 \
    --weight_aug 2.0 \
    --lambda_max 0.05 \
    --fisc_audio_feature_dim 32000 \
    --fisc_strength 1.5 \
    --edit_strength 1.5 \
    --feedback_every 25 \
    --feedback_mode clap_detac
```

To use reference-audio feedback, additionally specify:

```bash
--ref_audio_path /path/to/reference.wav
```

---

## Batch Personalized Editing

```bash
python run_batch_fis2c_personalized.py \
    --metadata /path/to/metadata.parquet \
    --audio_root /path/to/audio \
    --output_root ./outputs/personalized \
    --personalized_ckpt /path/to/personalized/checkpoint \
    --fisc_ckpt ./fisc_checkpoints/fisc_final.pt \
    --device cuda:0 \
    --validation_step 500 \
    --guidance_scale 30 \
    --weight_aug 2.0 \
    --lambda_max 0.05 \
    --fisc_audio_feature_dim 32000 \
    --fisc_strength 1.5 \
    --edit_strength 1.5 \
    --feedback_every 25 \
    --feedback_mode clap_detac
```

If a reference-audio column exists in the metadata, specify it with:

```bash
--ref_audio_key ref_audio_path
```

or use one shared reference file with:

```bash
--ref_audio_path /path/to/reference.wav
```

---

## Evaluation

The provided `test.py` can evaluate generated batch outputs.

Example:

```bash
python test.py \
    --batch_output_root ./outputs/zero_shot \
    --metadata /path/to/metadata.parquet \
    --audio_root /path/to/audio \
    --out_csv ./outputs/zero_shot/eval_metrics.csv \
    --metrics clap lpaps cqt detac
```

The evaluator supports loading samples from `batch_status.jsonl` and can also scan generated output folders when needed.

---

## Acknowledgement

FIS2C is built on top of **SteerMusic** and **AudioLDM2**. Personalized editing follows the SteerMusic+ setup based on personalized AudioLDM2 models. The feedback and evaluation pipeline additionally uses CLAP and MERT/deTAC components.

Please also follow the installation, model-license, and citation requirements of the original SteerMusic, AudioLDM2, DreamSound, CLAP, and MERT projects.
