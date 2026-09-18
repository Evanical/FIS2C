import argparse
import importlib.util
import math
import os
import re
import yaml
from pathlib import Path

import torch
from scipy.io.wavfile import write
from tqdm import tqdm

import fis2c_utils

from preprocessor import Preprocessor

# =========================
# Global GPU device
# =========================
DEVICE = None




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--prompt_ref", type=str, required=True")
    parser.add_argument("--concept", default="", type=str)
    parser.add_argument("--personalized_ckpt", default="", type=str)
    parser.add_argument("--output_dir", default="./FIS2C_output/", type=str)
    parser.add_argument("--guidance_scale", default=15.0, type=float)
    parser.add_argument("--weight_aug", default=2.0, type=float, help="Multiply guidance_scale")
    parser.add_argument("--fisc_strength", default=1.5, type=float, help="Scale FISC prompt compensation residual")
    parser.add_argument("--edit_strength", default=1.5, type=float, help="Scale DDS editing gradient")
    parser.add_argument("--validation_step", default=400, type=int)
    parser.add_argument("--add_reg", action="store_true")
    parser.add_argument("--lambd", default=0.05, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--fisc_ckpt", default="", type=str)
    parser.add_argument(
        "--no_fisc",
        action="store_true",
        help="Disable FISC completely for the personalized w/o FISC ablation.",
    )
    parser.add_argument("--lambda_max", default=0.05, type=float)
    parser.add_argument("--fisc_audio_feature_dim", default=32000, type=int)
    parser.add_argument("--ref_audio_path", default="", type=str)
    parser.add_argument("--config_yaml", default="config/autoencoder/16k_64.yaml", type=str)
    parser.add_argument("--audioldm2_path", default="", type=str, help="Local AudioLDM2 directory for offline loading.")
    parser.add_argument("--feedback_every", default=25, type=int, help="Decode and refresh external feedback every N optimization steps.")
    parser.add_argument("--feedback_eval_script", default="./test.py", type=str, help="Path to test.py containing CLAPScorer and DeTACScorer.")
    parser.add_argument("--feedback_device", default="cpu", type=str, help="Device used by CLAP/deTAC feedback models.")
    parser.add_argument("--clap_checkpoint_dir", default="clap/pretrained", type=str)
    parser.add_argument("--clap_checkpoint_name", default="music_audioset_epoch_15_esc_90.14.pt", type=str)
    parser.add_argument("--detac_model_name", default="m-a-p/MERT-v1-95M", type=str)
    parser.add_argument("--detac_layer", default=-1, type=int)
    parser.add_argument("--detac_quantile", default=0.1, type=float)
    parser.add_argument("--detac_cache_dir", default="./eval/.detac_mert_cache", type=str)
    parser.add_argument("--detac_max_audio_seconds", default=None, type=float)
    parser.add_argument("--detac_weight", default=1.0, type=float)
    parser.add_argument(
        "--feedback_mode",
        default="clap_detac",
        choices=[
            "only_clap",
            "only_detac",
            "clap_detac",
            "no_feedback",
            "alternate_clap_detac",
            "clap_then_detac",
            "detac_then_clap",
            "cycle_clap_detac_both",
        ],
        help=(
            "Personalized feedback schedule. "
            "alternate_clap_detac alternates CLAP-only and deTAC-only feedback per feedback round; "
            "clap_then_detac uses CLAP-only early and deTAC-only later; "
            "detac_then_clap uses deTAC-only early and CLAP-only later."
        ),
    )
    parser.add_argument(
        "--feedback_switch_every_rounds",
        default=1,
        type=int,
        help="For alternate_clap_detac/cycle_clap_detac_both: switch branch every N external-feedback refresh rounds.",
    )
    parser.add_argument(
        "--feedback_phase_ratio",
        default=0.5,
        type=float,
        help="For clap_then_detac/detac_then_clap: split point as ratio of validation_step.",
    )
    return parser.parse_args()


def make_personalized_prompts(prompt_ref, concept):
    match = re.search(r"\[(.*?)\]", prompt_ref)
    if not match:
        raise ValueError("prompt_ref must contain a concept enclosed in square brackets, e.g., [guitar]")
    prompt = re.sub(r"\[.*?\]", f"sks {concept}", prompt_ref)
    prompt_tgt = re.sub(r"\[.*?\]", f"{concept}", prompt_ref)
    return prompt, prompt_tgt


def save_audio(guidance_model, latent, audio_path, output_dir, suffix):
    audio = guidance_model.latents_to_audios(latent)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_{suffix}.wav")
    write(out_path, 16000, audio[0].detach().cpu().numpy())
    return out_path


def _load_feedback_eval_module(script_path):
    script_path = Path(script_path).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            f"feedback evaluator script not found: {script_path}. "
            "Pass --feedback_eval_script to the existing test.py file."
        )
    spec = importlib.util.spec_from_file_location("_steermusic_feedback_eval", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import feedback evaluator: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_float(value, default=0.0):
    try:
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _extract_detac_score(metrics):
    if isinstance(metrics, (int, float)):
        return _finite_float(metrics)
    if not isinstance(metrics, dict):
        raise TypeError(f"unexpected deTAC result type: {type(metrics)}")

    preferred_keys = [
        "detac", "deTAC", "detac_score", "deTAC_score", "tac_q",
        "TAC_q", "similarity", "score", "mean",
    ]
    for key in preferred_keys:
        if key in metrics:
            value = _finite_float(metrics[key], default=float("nan"))
            if math.isfinite(value):
                return value

    for value in metrics.values():
        value = _finite_float(value, default=float("nan"))
        if math.isfinite(value):
            return value
    raise ValueError(f"deTAC result contains no finite numeric score: {metrics}")


def _clamp_unit_score(value):
    return max(0.0, min(1.0, _finite_float(value)))


def _zero_feedback_indices(feedback, indices):
    """Zero selected entries in the 6-D feedback vector."""
    feedback = feedback.clone()
    flat = feedback.reshape(-1)
    for idx in indices:
        if idx < flat.numel():
            flat[idx] = 0.0
    return feedback


def resolve_active_feedback_mode(schedule_mode, step, total_steps, feedback_round, switch_every_rounds=1, phase_ratio=0.5):
    """
    Resolve high-level scheduled feedback into the active branch for this feedback refresh.

    Base modes:
      only_clap / only_detac / clap_detac / no_feedback

    Scheduled modes:
      alternate_clap_detac:
          feedback round 0 -> only_clap, round 1 -> only_detac, repeat.
      clap_then_detac:
          early DDS steps use only_clap, later DDS steps use only_detac.
      detac_then_clap:
          early DDS steps use only_detac, later DDS steps use only_clap.
      cycle_clap_detac_both:
          only_clap -> only_detac -> clap_detac, repeat.
    """
    mode = str(schedule_mode)
    if mode in ["only_clap", "only_detac", "clap_detac", "no_feedback"]:
        return mode

    switch_every_rounds = max(1, int(switch_every_rounds))
    block_id = int(feedback_round) // switch_every_rounds

    if mode == "alternate_clap_detac":
        return "only_clap" if block_id % 2 == 0 else "only_detac"

    if mode == "cycle_clap_detac_both":
        cycle = ["only_clap", "only_detac", "clap_detac"]
        return cycle[block_id % len(cycle)]

    split_step = int(float(total_steps) * float(phase_ratio))
    split_step = max(1, min(int(total_steps), split_step))

    if mode == "clap_then_detac":
        return "only_clap" if int(step) < split_step else "only_detac"

    if mode == "detac_then_clap":
        return "only_detac" if int(step) < split_step else "only_clap"

    raise ValueError(f"Unknown feedback_mode: {schedule_mode}")


def make_scheduled_feedback_tensor(
    active_mode,
    s_edit,
    s_pres,
    s_ref,
    previous_feedback,
    reference_mask,
    device,
    dtype,
):
    """
    Build the actual 6-D feedback vector consumed by FISC.

    The expected layout from steermusic_utils.make_feedback_tensor is:
      [edit, preserve, reference, delta_edit, delta_preserve, delta_reference]

    active_mode controls which branch is visible to FISC:
      only_clap  -> edit branch only
      only_detac -> source/reference deTAC branch only
      clap_detac -> edit + source/reference deTAC
      no_feedback -> zeros
    """
    active_mode = str(active_mode)

    if active_mode == "no_feedback":
        return torch.zeros_like(previous_feedback, device=device, dtype=dtype)

    keep_edit = active_mode in ["only_clap", "clap_detac"]
    keep_detac = active_mode in ["only_detac", "clap_detac"]

    s_edit_use = _finite_float(s_edit) if keep_edit else 0.0
    s_pres_use = _finite_float(s_pres) if keep_detac else 0.0
    s_ref_use = _finite_float(s_ref) if keep_detac else 0.0
    reference_mask_use = float(reference_mask) if keep_detac else 0.0

    current_feedback = steermusic_utils.make_feedback_tensor(
        s_edit=s_edit_use,
        s_pres=s_pres_use,
        s_ref=s_ref_use,
        prev_feedback=previous_feedback,
        reference_mask=reference_mask_use,
        device=device,
        dtype=dtype,
    )
    current_feedback = current_feedback.to(device=device, dtype=dtype)

    if active_mode == "only_clap":
        # Remove preservation/reference and their deltas.
        current_feedback = _zero_feedback_indices(current_feedback, [1, 2, 4, 5])
    elif active_mode == "only_detac":
        # Remove edit/reference? Keep reference deTAC if reference_mask is enabled.
        # We zero edit and delta_edit, but keep source/reference deTAC.
        current_feedback = _zero_feedback_indices(current_feedback, [0, 3])
        if reference_mask_use <= 0.0:
            current_feedback = _zero_feedback_indices(current_feedback, [2, 5])
    elif active_mode == "clap_detac":
        if reference_mask_use <= 0.0:
            current_feedback = _zero_feedback_indices(current_feedback, [2, 5])
    else:
        raise ValueError(f"Unknown active feedback mode: {active_mode}")

    return current_feedback.detach()


def feedback_selection_score(active_mode, current_feedback, previous_feedback, reference_mask, detac_weight):
    """Best-latent selection score consistent with the active feedback branch."""
    try:
        over = steermusic_utils.over_edit_penalty(current_feedback, previous_feedback)
        over_value = _finite_float(over)
    except Exception:
        over_value = 0.0

    active_mode = str(active_mode)
    edit_term = _finite_float(current_feedback[0]) if active_mode in ["only_clap", "clap_detac"] else 0.0
    detac_term = 0.0
    if active_mode in ["only_detac", "clap_detac"]:
        detac_term = _finite_float(current_feedback[1])
        if float(reference_mask) > 0.0 and current_feedback.numel() > 2:
            detac_term += _finite_float(current_feedback[2])

    return edit_term + float(detac_weight) * detac_term - over_value


def _write_feedback_audio(guidance_model, latent, path):
    with torch.no_grad():
        audio = guidance_model.latents_to_audios(latent.detach())
    write(str(path), 16000, audio[0].detach().float().cpu().numpy())


def _build_feedback_scorers(opt, need_detac):
    module = _load_feedback_eval_module(opt.feedback_eval_script)
    if not hasattr(module, "CLAPScorer"):
        raise AttributeError("feedback evaluator does not define CLAPScorer")

    clap = module.CLAPScorer(
        opt.feedback_device,
        opt.clap_checkpoint_dir,
        opt.clap_checkpoint_name,
    )

    detac = None
    if need_detac:
        if not hasattr(module, "DeTACScorer"):
            raise AttributeError("feedback evaluator does not define DeTACScorer")
        detac = module.DeTACScorer(
            device=opt.feedback_device,
            model_name=opt.detac_model_name,
            layer=opt.detac_layer,
            quantile=opt.detac_quantile,
            cache_dir=opt.detac_cache_dir,
            max_audio_seconds=opt.detac_max_audio_seconds,
        )
    return clap, detac


def _score_external_feedback(
    guidance_model,
    latent,
    output_dir,
    step,
    target_prompt,
    source_audio_path,
    reference_audio_path,
    clap_scorer,
    detac_scorer,
):
    temp_path = Path(output_dir) / f".__feedback_step_{int(step):05d}.wav"
    _write_feedback_audio(guidance_model, latent, temp_path)
    try:
        s_edit = _clamp_unit_score(
            clap_scorer.score_audio_text(str(temp_path), target_prompt)
        )
        s_pres = 0.0
        s_ref = 0.0
        if detac_scorer is not None:
            s_pres = _clamp_unit_score(
                _extract_detac_score(
                    detac_scorer.score_pair(source_audio_path, str(temp_path))
                )
            )
            if reference_audio_path:
                s_ref = _clamp_unit_score(
                    _extract_detac_score(
                        detac_scorer.score_pair(reference_audio_path, str(temp_path))
                    )
                )
        return s_edit, s_pres, s_ref
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

def encode_audio_path(preprocessor, guidance_model, audio_path, device):
    """
    Use exactly the same preprocessing path as source audio.
    Force tensors onto GPU device.
    """

    log_mel_spec, _, _, _ = preprocessor.read_audio_file(
        filename=audio_path
    )

    log_mel_spec = (
        log_mel_spec
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    )

    with torch.no_grad():
        latent = guidance_model.encode_audio(log_mel_spec)

    latent = latent.to(device=device, non_blocking=True)

    return latent


def _load_fisc_state_for_dim(ckpt_path, map_location="cpu"):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "fisc_state_dict" in ckpt:
            state = ckpt["fisc_state_dict"]
        elif "fisc" in ckpt:
            state = ckpt["fisc"]
        else:
            state = ckpt
    else:
        state = ckpt
    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("fisc."):
            k = k[len("fisc."):]
        clean[k] = v
    return clean


def infer_fisc_audio_dim_from_ckpt(ckpt_path, fallback_dim=32000):
    fallback_dim = int(fallback_dim)
    state = _load_fisc_state_for_dim(ckpt_path, map_location="cpu")
    if not state:
        return fallback_dim
    for key in ["source_audio_projector.net.0.weight", "reference_audio_projector.net.0.weight"]:
        w = state.get(key, None)
        if hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[1])
    return fallback_dim


def build_fisc_model(opt):
    ckpt_dim = infer_fisc_audio_dim_from_ckpt(opt.fisc_ckpt, fallback_dim=opt.fisc_audio_feature_dim)
    if ckpt_dim != int(opt.fisc_audio_feature_dim):
        print(
            "[WARN] --fisc_audio_feature_dim does not match the checkpoint: "
            f"requested={opt.fisc_audio_feature_dim}, "
            f"checkpoint_projector_in_features={ckpt_dim}. "
            f"Using the checkpoint dimension ({ckpt_dim}) for this run."
        )
    else:
        print("[INFO] checkpoint FISC audio feature dim:", ckpt_dim)

    try:
        fisc = steermusic_utils.FISCModel(
            lambda_max=opt.lambda_max,
            audio_feature_dim=int(ckpt_dim),
        ).to(DEVICE)
    except TypeError:
        raise TypeError(
            "The current steermusic_utils.FISCModel does not support "
            "the audio_feature_dim argument. Please ensure that "
            "steermusic_utils.py explicitly initializes the source audio "
            "projector with a non-LazyLinear layer."
        )
    fisc.audio_feature_dim = int(ckpt_dim)
    return fisc, int(ckpt_dim)


def main():
    global DEVICE

    opt = parse_args()

    # Fixed ablation variant: clap_detac.
    opt.no_fisc = False
    if not opt.fisc_ckpt:
        raise ValueError("--fisc_ckpt is required for the clap_detac ablation.")
    if int(opt.feedback_every) <= 0:
        raise ValueError("--feedback_every must be positive")

    DEVICE = torch.device(
        opt.device if torch.cuda.is_available() else "cpu"
    )

    print("[INFO] using device:", DEVICE)
    print("[INFO] cuda available:", torch.cuda.is_available())

    if "cuda" in str(DEVICE):
        torch.cuda.set_device(DEVICE)

    if getattr(opt, "audioldm2_path", ""):
        os.environ["AUDIOLDM2_LOCAL_PATH"] = os.path.abspath(os.path.expanduser(opt.audioldm2_path))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"
        print("[INFO] using local AudioLDM2 path:", os.environ["AUDIOLDM2_LOCAL_PATH"])

    os.makedirs(opt.output_dir, exist_ok=True)

    prompt, prompt_tgt = make_personalized_prompts(opt.prompt_ref, opt.concept)
    print("[INFO] Target prompt:", prompt)
    print("[INFO] guidance_scale:", opt.guidance_scale)
    print("[INFO] weight_aug:", opt.weight_aug)
    print("[INFO] effective_guidance_scale:", opt.guidance_scale * opt.weight_aug)
    print("[INFO] fisc_strength:", opt.fisc_strength)
    print("[INFO] edit_strength:", opt.edit_strength)
    print("[INFO] no_fisc:", opt.no_fisc)

    print("[INFO] feedback ablation:", "scheduled personalized feedback")
    print("[INFO] feedback_mode:", opt.feedback_mode)
    print("[INFO] feedback_every:", opt.feedback_every)
    print("[INFO] feedback_switch_every_rounds:", opt.feedback_switch_every_rounds)
    print("[INFO] feedback_phase_ratio:", opt.feedback_phase_ratio)
    print("[INFO] feedback_device:", opt.feedback_device)

    need_detac_feedback = str(opt.feedback_mode) not in ["only_clap", "no_feedback"]
    clap_scorer, detac_scorer = _build_feedback_scorers(
        opt,
        need_detac=need_detac_feedback,
    )

    config = yaml.load(open(opt.config_yaml, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)

    # Load the AudioLDM2 base model.
    # opt.personalized_ckpt is a .pt checkpoint, not a HuggingFace repo/path.
    # AudioLDM2_pipe / from_pretrained must receive the AudioLDM2 base directory.
    base_hf_key = opt.audioldm2_path if opt.audioldm2_path else None

    if base_hf_key:
        guidance_model = steermusic_utils.AudioLDM2_pipe(
            DEVICE,
            fp16=False,
            vram_O=False,
            hf_key=base_hf_key,
        )

        personalized_model = steermusic_utils.AudioLDM2_pipe(
            DEVICE,
            fp16=False,
            vram_O=False,
            hf_key=base_hf_key,
        )
    else:
        guidance_model = steermusic_utils.AudioLDM2_pipe(
            DEVICE,
            fp16=False,
            vram_O=False,
        )

        personalized_model = steermusic_utils.AudioLDM2_pipe(
            DEVICE,
            fp16=False,
            vram_O=False,
        )

    # Then load the personalized .pt weights into personalized_model.
    if opt.personalized_ckpt:
        if os.path.isfile(opt.personalized_ckpt):
            print("[INFO] loading personalized weights:", opt.personalized_ckpt)
            ckpt = torch.load(opt.personalized_ckpt, map_location=DEVICE)

            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    state = ckpt["state_dict"]
                elif "model_state_dict" in ckpt:
                    state = ckpt["model_state_dict"]
                elif "personalized_state_dict" in ckpt:
                    state = ckpt["personalized_state_dict"]
                else:
                    state = ckpt
            else:
                state = ckpt

            clean_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                if k.startswith("model."):
                    k = k[len("model."):]
                clean_state[k] = v

            try:
                missing, unexpected = personalized_model.load_state_dict(clean_state, strict=False)
                print(f"[INFO] personalized weights loaded; missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception as e:
                print("[WARN] direct load_state_dict failed:", repr(e))
                if hasattr(personalized_model, "pipe"):
                    missing, unexpected = personalized_model.pipe.load_state_dict(clean_state, strict=False)
                    print(f"[INFO] personalized weights loaded into personalized_model.pipe; missing={len(missing)}, unexpected={len(unexpected)}")
                elif hasattr(personalized_model, "unet"):
                    missing, unexpected = personalized_model.unet.load_state_dict(clean_state, strict=False)
                    print(f"[INFO] personalized weights loaded into personalized_model.unet; missing={len(missing)}, unexpected={len(unexpected)}")
                else:
                    raise
        else:
            print("[WARN] --personalized_ckpt is not a local file, skipped loading personalized .pt weights:", opt.personalized_ckpt)

    guidance_model.eval()
    personalized_model.eval()

    for p in guidance_model.parameters():
        p.requires_grad = False
    for p in personalized_model.parameters():
        p.requires_grad = False

    log_mel_spec, _, _, _ = preprocessor.read_audio_file(filename=opt.audio_path)
    log_mel_spec = log_mel_spec.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        source_latent = guidance_model.encode_audio(log_mel_spec)
        latent = source_latent.clone().detach().requires_grad_(True)

        c_s, c_s_gen, mask_s = guidance_model.get_text_embeds(opt.prompt_ref)
        c_t, c_t_gen, mask_t = personalized_model.get_text_embeds(prompt)
        c_t_plain, c_t_plain_gen, mask_t_plain = guidance_model.get_text_embeds(prompt_tgt)

    use_fisc = (
        not opt.no_fisc
        and bool(opt.fisc_ckpt)
        and float(opt.lambda_max) > 0.0
    )

    fisc = None
    fisc_feature_dim = int(opt.fisc_audio_feature_dim)
    ref_latent = None
    reference_feature = None
    reference_mask = 0.0

    if use_fisc:
        print("[INFO] FISC enabled for personalized editing.")
        fisc, fisc_feature_dim = build_fisc_model(opt)
        fisc.eval()
        print("[INFO] effective FISC audio feature dim:", fisc_feature_dim)

        if opt.ref_audio_path:
            ref_latent = encode_audio_path(
                preprocessor,
                guidance_model,
                opt.ref_audio_path,
                DEVICE,
            )
            reference_feature = steermusic_utils.normalize_fisc_audio_feature(
                ref_latent.detach(),
                target_dim=fisc_feature_dim,
            )
            reference_mask = 1.0
            print("[INFO] using reference audio:", opt.ref_audio_path)
        else:
            print("[INFO] no --ref_audio_path provided; reference branch disabled")

        source_audio_feature0 = steermusic_utils.normalize_fisc_audio_feature(
            source_latent.detach(),
            target_dim=fisc_feature_dim,
        )
        feedback0 = torch.zeros(6, device=DEVICE, dtype=source_latent.dtype)
        t0 = torch.tensor(
            [guidance_model.min_step],
            device=DEVICE,
            dtype=torch.long,
        )

        with torch.no_grad():
            _ = fisc(
                target_prompt_embeds=c_t,
                target_generated_embeds=c_t_gen,
                source_prompt_embeds=c_s,
                source_generated_embeds=c_s_gen,
                source_audio_feature=source_audio_feature0,
                feedback=feedback0,
                timestep=t0,
                reference_feature=reference_feature,
                reference_mask=reference_mask,
            )

        steermusic_utils.load_fisc_checkpoint(
            fisc,
            opt.fisc_ckpt,
            strict=False,
            map_location=DEVICE,
        )
    else:
        print(
            "[INFO] FISC disabled: running personalized w/o FISC with "
            "the original personalized target embeddings."
        )
        if opt.ref_audio_path:
            print(
                "[INFO] --ref_audio_path is ignored because the reference "
                "adapter belongs to FISC."
            )

    optim = torch.optim.SGD([latent], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.9)

    previous_feedback = torch.zeros(6, device=DEVICE, dtype=source_latent.dtype)
    best_latent = latent.detach().clone()
    best_score = -1e9

    effective_guidance_scale = opt.guidance_scale * opt.weight_aug

    print("[INFO] FISC personalized editing with scheduled feedback:", opt.feedback_mode)
    feedback_round = 0
    for i in tqdm(range(opt.validation_step + 1)):
        optim.zero_grad()
        x = latent

        
        t = torch.randint(
            personalized_model.min_step,
            personalized_model.max_step + 1,
            [x.shape[0]],
            dtype=torch.long,
            device=DEVICE,
        )
        noise = torch.randn_like(x)

        if use_fisc:
            source_audio_feature = steermusic_utils.normalize_fisc_audio_feature(
                source_latent.detach(),
                target_dim=fisc_feature_dim,
            )

            # Only a real reference audio enables the FISC reference adapter.
            out = fisc(
                target_prompt_embeds=c_t,
                target_generated_embeds=c_t_gen,
                source_prompt_embeds=c_s,
                source_generated_embeds=c_s_gen,
                source_audio_feature=source_audio_feature,
                feedback=previous_feedback,
                timestep=t,
                reference_feature=reference_feature,
                reference_mask=reference_mask,
            )

            target_prompt_embeds_use = out["prompt_embeds"]
            target_generated_embeds_use = out["generated_prompt_embeds"]

            # Rescale the FISC residual around the original personalized target
            # embeddings without changing the personalized model itself.
            if float(opt.fisc_strength) != 1.0:
                target_prompt_embeds_use = (
                    c_t
                    + float(opt.fisc_strength)
                    * (target_prompt_embeds_use - c_t)
                )
                target_generated_embeds_use = (
                    c_t_gen
                    + float(opt.fisc_strength)
                    * (target_generated_embeds_use - c_t_gen)
                )

            if i % 50 == 0:
                with torch.no_grad():
                    delta_prompt = out.get("delta_prompt", None)
                    lambda_prompt = out.get("lambda_prompt", None)
                    if delta_prompt is not None:
                        delta_prompt_norm = delta_prompt.float().norm().item()
                        target_prompt_norm = c_t.float().norm().item()
                        delta_ratio = delta_prompt_norm / max(target_prompt_norm, 1e-6)
                    else:
                        delta_prompt_norm = 0.0
                        target_prompt_norm = c_t.float().norm().item()
                        delta_ratio = 0.0
                    lambda_mean = (
                        lambda_prompt.float().mean().item()
                        if lambda_prompt is not None
                        else 0.0
                    )
                    print(
                        "[FISC]",
                        f"step={i}",
                        f"lambda_mean={lambda_mean:.6f}",
                        f"delta_prompt_norm={delta_prompt_norm:.6f}",
                        f"target_prompt_norm={target_prompt_norm:.6f}",
                        f"delta_ratio={delta_ratio:.6f}",
                        f"reference_mask={reference_mask:.1f}",
                    )
        else:
            # Clean w/o FISC ablation:
            # keep the personalized target model and all DDS/reg terms,
            # but bypass FISC source projection, feedback gate, prompt
            # compensation, and reference semantic adapter.
            target_prompt_embeds_use = c_t
            target_generated_embeds_use = c_t_gen

        noise_pred, _, _ = personalized_model.predict_noise(
            prompt_embds=target_prompt_embeds_use,
            generated_prompt_embds=target_generated_embeds_use,
            attention_mask=mask_t,
            mel_spec=x,
            guidance_scale=effective_guidance_scale,
            as_latent=True,
            t=t,
            noise=noise,
        )

        with torch.no_grad():
            noise_pred_src, _, _ = guidance_model.predict_noise(
                prompt_embds=c_s,
                generated_prompt_embds=c_s_gen,
                attention_mask=mask_s,
                mel_spec=source_latent,
                guidance_scale=effective_guidance_scale,
                as_latent=True,
                t=t,
                noise=noise,
            )

        w = 1 - personalized_model.alphas[t]
        grad = opt.edit_strength * torch.nan_to_num(w * (noise_pred - noise_pred_src))

        if opt.add_reg:
            
            with torch.no_grad():
                noise_pred_phi0, _, _ = guidance_model.predict_noise(
                    prompt_embds=c_t_plain,
                    generated_prompt_embds=c_t_plain_gen,
                    attention_mask=mask_t_plain,
                    mel_spec=x,
                    guidance_scale=effective_guidance_scale,
                    as_latent=True,
                    t=t,
                    noise=noise,
                )
            grad = grad + torch.nan_to_num(opt.lambd * w * (noise_pred - noise_pred_phi0))

        loss = steermusic_utils.SpecifyGradient.apply(x, grad)
        loss.backward()
        optim.step()
        scheduler.step()

        refresh_feedback = (
            i % int(opt.feedback_every) == 0
            or i == int(opt.validation_step)
        )
        if refresh_feedback:
            with torch.no_grad():
                s_edit, s_pres, s_ref = _score_external_feedback(
                    guidance_model=guidance_model,
                    latent=latent,
                    output_dir=opt.output_dir,
                    step=i,
                    target_prompt=prompt_tgt,
                    source_audio_path=opt.audio_path,
                    reference_audio_path=opt.ref_audio_path,
                    clap_scorer=clap_scorer,
                    detac_scorer=detac_scorer,
                )
                active_feedback_mode = resolve_active_feedback_mode(
                    schedule_mode=opt.feedback_mode,
                    step=i,
                    total_steps=opt.validation_step,
                    feedback_round=feedback_round,
                    switch_every_rounds=opt.feedback_switch_every_rounds,
                    phase_ratio=opt.feedback_phase_ratio,
                )
                current_feedback = make_scheduled_feedback_tensor(
                    active_mode=active_feedback_mode,
                    s_edit=s_edit,
                    s_pres=s_pres,
                    s_ref=s_ref,
                    previous_feedback=previous_feedback,
                    reference_mask=reference_mask,
                    device=DEVICE,
                    dtype=source_latent.dtype,
                )
                score = feedback_selection_score(
                    active_mode=active_feedback_mode,
                    current_feedback=current_feedback,
                    previous_feedback=previous_feedback,
                    reference_mask=reference_mask,
                    detac_weight=opt.detac_weight,
                )
                print(
                    "[FEEDBACK]",
                    f"schedule={opt.feedback_mode}",
                    f"active={active_feedback_mode}",
                    f"round={feedback_round}",
                    f"step={i}",
                    f"clap_raw={s_edit:.6f}",
                    f"detac_source_raw={s_pres:.6f}",
                    f"detac_reference_raw={s_ref:.6f}",
                    f"score={float(score):.6f}",
                    "vector=" + ",".join(f"{v:.6f}" for v in current_feedback.detach().float().reshape(-1).cpu().tolist()),
                )
                if float(score) > best_score:
                    best_score = float(score)
                    best_latent = latent.detach().clone()
                previous_feedback = current_feedback.detach()
                feedback_round += 1

    safe_mode = re.sub(r"[^A-Za-z0-9._-]+", "_", str(opt.feedback_mode)).strip("_") or "scheduled"
    suffix_best = f"personalized_{safe_mode}_best"
    suffix_last = f"personalized_{safe_mode}_last"
    best_path = save_audio(
        guidance_model,
        best_latent,
        opt.audio_path,
        opt.output_dir,
        suffix_best,
    )
    last_path = save_audio(
        guidance_model,
        latent,
        opt.audio_path,
        opt.output_dir,
        suffix_last,
    )
    print("[INFO] Saved best:", best_path)
    print("[INFO] Saved last:", last_path)


if __name__ == "__main__":
    main()
