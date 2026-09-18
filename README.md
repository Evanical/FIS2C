# FIS2C

Official implementation of **FIS2C** for feedback-guided zero-shot text-guided and personalized music editing.  
FIS2C is built on top of [SteerMusic](https://github.com/sony/steermusic) and uses AudioLDM2 as the pretrained diffusion backbone.

---

## 1. Environment Setup

FIS2C follows the environment used by SteerMusic. The original SteerMusic implementation was tested with:

- Python 3.8.10
- PyTorch 2.2.0 + CUDA 12.1
- AudioLDM2

Create the environment:

```bash
conda create -n fis2c python=3.8.10
conda activate fis2c
```

Install PyTorch:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you encounter

```text
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

follow the compatibility fix used by SteerMusic and replace `cached_download` with `hf_hub_download` in:

```text
diffusers/utils/dynamic_modules_utils.py
```

---

## 2. Required Pretrained Models

Large pretrained model weights are **not included in this repository**. Please obtain them from the corresponding official sources.

### 2.1 AudioLDM2

FIS2C uses the same pretrained AudioLDM2 backbone as SteerMusic.

Official model:

```text
https://huggingface.co/cvssp/audioldm2
```

The model can be downloaded automatically by Diffusers when internet access is available.

For offline inference, download the model locally and specify:

```bash
--audioldm2_path /path/to/audioldm2
```

Example directory:

```text
pretrained_models/
└── audioldm2/
    ├── feature_extractor/
    ├── language_model/
    ├── projection_model/
    ├── scheduler/
    ├── text_encoder/
    ├── text_encoder_2/
    ├── tokenizer/
    ├── tokenizer_2/
    ├── unet/
    ├── vae/
    └── vocoder/
```

### 2.2 CLAP

CLAP is used for semantic feedback and objective evaluation during inference.

Official implementation:

```text
https://github.com/LAION-AI/CLAP
```

The current FIS2C scripts expect the following checkpoint:

```text
music_audioset_epoch_15_esc_90.14.pt
```

Place the checkpoint under:

```text
clap/
└── pretrained/
    └── music_audioset_epoch_15_esc_90.14.pt
```

The corresponding default arguments are:

```text
--clap_checkpoint_dir clap/pretrained
--clap_checkpoint_name music_audioset_epoch_15_esc_90.14.pt
```

The checkpoint itself is not distributed in this repository.

### 2.3 MERT / deTAC

FIS2C uses MERT features for deTAC-based structural feedback and evaluation.

Default model:

```text
m-a-p/MERT-v1-95M
```

Official Hugging Face model:

```text
https://huggingface.co/m-a-p/MERT-v1-95M
```

When internet access is available, Transformers downloads the model automatically on first use.

For offline experiments, download the model beforehand and make sure it is available in the local Hugging Face cache.

### 2.4 FIS2C Controller Checkpoint

The FIS2C controller checkpoint is produced by the training script in this repository.

After training, the final checkpoint is saved as:

```text
fisc_checkpoints/fisc_final.pt
```

Use it for inference with:

```bash
--fisc_ckpt /path/to/fisc_final.pt
```

The source repository does not require the checkpoint to run training from scratch. A pretrained FIS2C controller checkpoint can be released separately for direct reproduction of inference results.

### 2.5 Personalized AudioLDM2 Checkpoint

Personalized FIS2C editing follows the SteerMusic+ setup and requires a personalized AudioLDM2 model.

SteerMusic:

```text
https://github.com/sony/steermusic
```

DreamSound personalization code:

```text
https://github.com/zelaki/DreamSound
```

Follow the DreamSound AudioLDM2 DreamBooth procedure to obtain a personalized checkpoint, then pass it to FIS2C using:

```bash
--personalized_ckpt /path/to/personalized/checkpoint
```

The personalized diffusion checkpoint is not included in this repository.

---

## 3. Dataset Preparation

### ZoME-Bench

FIS2C is trained using ZoME-Bench metadata.

The raw ZoME-Bench audio is **not redistributed in this repository**. Please obtain the dataset/audio according to the official ZoME-Bench release and prepare the corresponding audio clips locally.

The training metadata should contain at least:

```text
original_prompt
editing_prompt
audio_path
```

Supported metadata formats:

```text
.parquet
.csv
.json
.jsonl
.ndjson
```

A typical local layout is:

```text
ZoME-Bench/
├── metadata_with_audio.parquet
└── audio/
    ├── xxx_0_10.wav
    ├── xxx_10_20.wav
    └── ...
```

If `audio_path` is relative, provide the audio directory through:

```bash
--audio_root /path/to/ZoME-Bench/audio
```

If the metadata contains `ytid`, `start_s`, and `end_s`, the loader also searches for audio files using:

```text
{ytid}_{start_s}_{end_s}.wav
```

If paper-specific train/validation/test split files are provided in this repository, they contain metadata only and do not include the original audio.

---

## 4. Training FIS2C

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

The final checkpoint is saved to:

```text
./fisc_checkpoints/fisc_final.pt
```

For a quick smoke test:

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

During training, semantic and structural feedback terms are implemented using differentiable score-level surrogates at sampled DDS states. CLAP and deTAC are used for feedback-based inference and evaluation rather than being directly back-propagated through during controller training.

---

## 5. Zero-shot Text-guided Music Editing

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

Available feedback modes:

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

## 6. Batch Zero-shot Editing

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

## 7. Personalized Music Editing

Personalized FIS2C editing requires a personalized AudioLDM2 checkpoint prepared following the SteerMusic+/DreamSound setup.

The source prompt must contain the edited concept inside square brackets.

Example:

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
    --lambda_max 0.03 \
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

## 8. Batch Personalized Editing

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
    --lambda_max 0.03 \
    --fisc_audio_feature_dim 32000 \
    --fisc_strength 1.5 \
    --edit_strength 1.5 \
    --feedback_every 25 \
    --feedback_mode clap_detac
```

If a reference-audio column exists in the metadata, specify:

```bash
--ref_audio_key ref_audio_path
```

or provide one shared reference file with:

```bash
--ref_audio_path /path/to/reference.wav
```

---

## 9. Evaluation

The provided `test.py` evaluates generated batch outputs.

Example:

```bash
python test.py \
    --batch_output_root ./outputs/zero_shot \
    --metadata /path/to/metadata.parquet \
    --audio_root /path/to/audio \
    --out_csv ./outputs/zero_shot/eval_metrics.csv \
    --metrics clap lpaps cqt detac
```

The evaluator can use `batch_status.jsonl` or scan generated output folders directly.

Before running evaluation, make sure the required CLAP and MERT/deTAC resources described in Sec. 2 are available.

---

## 10. Recommended Repository Layout

```text
FIS2C/
├── README.md
├── requirements.txt
├── .gitignore
│
├── train_fis2c.py
├── fis2c_edit.py
├── fis2c_personalized_edit.py
├── steermusic_utils.py
├── preprocessor.py
├── test.py
│
├── run_batch_fis2c_edit.py
├── run_batch_fis2c_personalized.py
│
├── config/
│   └── autoencoder/
│       └── 16k_64.yaml
│
├── eval/
│   ├── CDPAM.py
│   ├── CQT1_PCC.py
│   ├── clap_score.py
│   ├── deTAC.py
│   ├── lpaps.py
│   ├── lpaps_score.py
│   ├── meta_clap_consistency.py
│   ├── pretrained_networks.py
│   └── utils.py
│
└── clap/
    └── pretrained/
        └── music_audioset_epoch_15_esc_90.14.pt   # download separately
```

The following are intentionally not included in the source repository:

```text
ZoME-Bench raw audio
AudioLDM2 pretrained weights
CLAP pretrained checkpoint
MERT model weights
FIS2C training checkpoints
personalized AudioLDM2 checkpoints
training/inference output folders
evaluation caches
```

---

## 11. Reproducing From a Fresh Clone

Clone the repository:

```bash
git clone https://github.com/Evanical/FIS2C.git
cd FIS2C
```

Create the environment and install dependencies:

```bash
conda create -n fis2c python=3.8.10
conda activate fis2c

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Then prepare:

1. AudioLDM2;
2. the CLAP checkpoint;
3. MERT/deTAC;
4. ZoME-Bench metadata and audio;
5. a FIS2C checkpoint, either trained from scratch or downloaded separately;
6. a personalized AudioLDM2 checkpoint only when running personalized editing.

After all required resources are prepared, follow Secs. 4--9 for training, inference, and evaluation.

---

## 12. Upstream Projects

FIS2C builds on the following open-source projects:

- SteerMusic: https://github.com/sony/steermusic
- AudioLDM2: https://huggingface.co/cvssp/audioldm2
- LAION-CLAP: https://github.com/LAION-AI/CLAP
- MERT: https://huggingface.co/m-a-p/MERT-v1-95M
- DreamSound: https://github.com/zelaki/DreamSound

Please follow the licenses and citation requirements of the corresponding upstream projects when using their code or pretrained models.

---

## 13. Acknowledgement

FIS2C is built on top of **SteerMusic** and **AudioLDM2**. Personalized editing follows the **SteerMusic+** setup based on personalized AudioLDM2 models. The feedback and evaluation pipeline additionally uses **CLAP** and **MERT/deTAC**.

Please also follow the installation, licensing, and citation requirements of SteerMusic, AudioLDM2, DreamSound, CLAP, and MERT.
