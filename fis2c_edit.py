import argparse
import inspect
import os
import yaml

import torch
import torch.nn.functional as F
from scipy.io.wavfile import write
from tqdm import tqdm

import fis2c_utils

from preprocessor import Preprocessor



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feedback_mode",
        type=str,
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
            "FISC feedback-module ablation / schedule mode. "
            "only_clap keeps only the edit/CLAP branch; "
            "only_detac keeps only the preserve/deTAC branch; "
            "clap_detac keeps edit/CLAP + preserve/deTAC branches; "
            "alternate_clap_detac alternates only_clap and only_detac every N steps; "
            "clap_then_detac uses only_clap in the early phase and only_detac later; "
            "detac_then_clap does the reverse; "
            "cycle_clap_detac_both cycles only_clap -> only_detac -> clap_detac; "
            "no_feedback feeds a zero feedback vector to FISC."
        ),
    )

    parser.add_argument(
        "--feedback_switch_every",
        type=int,
        default=25,
        help=(
            "For alternate_clap_detac / cycle_clap_detac_both, switch active feedback branch every N DDS steps. "
            "Example: 25 means steps 0-24 use CLAP, 25-49 use deTAC, etc."
        ),
    )
    parser.add_argument(
        "--feedback_phase_ratio",
        type=float,
        default=0.5,
        help=(
            "For clap_then_detac / detac_then_clap, split point as a fraction of validation_step. "
            "0.4 means the first 40% of steps use the first branch and the rest use the second branch."
        ),
    )

    
    parser.add_argument(
        "--audio_path",
        "--source_audio_path",
        dest="audio_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prompt_ref",
        "--source_prompt",
        dest="prompt_ref",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prompt",
        "--target_prompt",
        dest="prompt",
        type=str,
        required=True,
    )

    parser.add_argument("--output_dir", default="./FISC_SteerMusic_output/", type=str)
    parser.add_argument("--validation_step", default=400, type=int)
    parser.add_argument("--guidance_scale", default=30.0, type=float)
    parser.add_argument("--weight_aug", default=2.0, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--fisc_ckpt", default="", type=str)
    parser.add_argument("--lambda_max", default=0.05, type=float)
    parser.add_argument("--fisc_audio_feature_dim", default=32000, type=int)
    parser.add_argument("--config_yaml", default="config/autoencoder/16k_64.yaml", type=str)
    parser.add_argument(
        "--fisc_strength",
        default=1.3,
        type=float,
        help="FISC prompt residual strength. Try 1.5 or 2.0 if FISC effect is too weak.",
    )
    parser.add_argument(
        "--edit_strength",
        default=1.3,
        type=float,
        help="DDS gradient multiplier. Try 1.5 or 2.0 if edit is too weak.",
    )

    return parser.parse_args()



def normalize_fisc_audio_feature(audio_feature, target_dim=32000, eps=1e-6, detach=True):

    if audio_feature is None:
        return None

    if detach:
        audio_feature = audio_feature.detach()

    if audio_feature.dim() == 1:
        audio_feature = audio_feature.unsqueeze(0)

    audio_feature = audio_feature.float()
    audio_feature = audio_feature.flatten(1)

    current_dim = int(audio_feature.shape[1])

    if current_dim < int(target_dim):
        audio_feature = F.pad(audio_feature, (0, int(target_dim) - current_dim), value=0.0)
    elif current_dim > int(target_dim):
        audio_feature = audio_feature[:, : int(target_dim)]

    mean = audio_feature.mean(dim=1, keepdim=True)
    std = audio_feature.std(dim=1, keepdim=True).clamp_min(eps)
    audio_feature = (audio_feature - mean) / std
    audio_feature = torch.nan_to_num(audio_feature, nan=0.0, posinf=0.0, neginf=0.0)

    return audio_feature


def save_audio(guidance_model, latent, audio_path, output_dir, suffix):
    audio = guidance_model.latents_to_audios(latent)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_{suffix}.wav")
    write(out_path, 16000, audio[0].detach().cpu().numpy())
    return out_path


def _load_fisc_state_for_dim(ckpt_path, map_location="cpu"):
    """读取 checkpoint 中的 FISC state_dict，用来推断 projector 输入维度。"""
    if not ckpt_path:
        return None
    if not os.path.exists(ckpt_path):
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

    for key in [
        "source_audio_projector.net.0.weight",
        "reference_audio_projector.net.0.weight",
    ]:
        w = state.get(key, None)
        if hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[1])

    return fallback_dim


def _first_linear_in_features(module):

    if module is None:
        return None
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            in_features = int(getattr(m, "in_features", 0))
            if in_features > 0:
                return in_features
    return None


def get_fisc_audio_feature_dim(fisc, requested_dim):

    requested_dim = int(requested_dim)
    actual_dim = _first_linear_in_features(getattr(fisc, "source_audio_projector", None))
    if actual_dim is None or actual_dim <= 0:
        actual_dim = int(getattr(fisc, "audio_feature_dim", requested_dim))

    actual_dim = int(actual_dim)
    if actual_dim != requested_dim:
        raise RuntimeError(
            "FISC audio feature dim mismatch before checkpoint loading: "
            f"requested={requested_dim}, projector_in_features={actual_dim}. "
            
        )
    return actual_dim


def build_fisc_model(opt):
    if not hasattr(steermusic_utils, "FISCModel"):
        raise AttributeError(
            "can't find FISCModel"
        )

    ckpt_dim = infer_fisc_audio_dim_from_ckpt(opt.fisc_ckpt, fallback_dim=opt.fisc_audio_feature_dim)
    requested_dim = int(opt.fisc_audio_feature_dim)
    if ckpt_dim != requested_dim:
        print(
            "[WARN] --fisc_audio_feature_dim does not match checkpoint"
            f" requested={requested_dim}, checkpoint_projector_in_features={ckpt_dim}。"
        )
    else:
        print("[INFO] checkpoint FISC audio feature dim:", ckpt_dim)

    kwargs = {"lambda_max": opt.lambda_max}
    try:
        sig = inspect.signature(steermusic_utils.FISCModel.__init__)
        if "audio_feature_dim" in sig.parameters:
            kwargs["audio_feature_dim"] = ckpt_dim
        elif "source_audio_dim" in sig.parameters:
            kwargs["source_audio_dim"] = ckpt_dim
        elif "input_dim" in sig.parameters:
            kwargs["input_dim"] = ckpt_dim
    except Exception:
        pass

    try:
        fisc = steermusic_utils.FISCModel(**kwargs).to(opt.device)
    except TypeError:
        raise TypeError(
            "fis2c_utils.FISCModel does not support audio_feature_dim/source_audio_dim/input_dim。\n"
        )

    fisc.audio_feature_dim = int(ckpt_dim)
    actual_dim = get_fisc_audio_feature_dim(fisc, ckpt_dim)
    print("[INFO] built FISC projector input dim:", actual_dim)
    return fisc, int(ckpt_dim)





def _safe_float(x, default=0.0):
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float(default)


def _zero_feedback_indices(feedback, indices):
    """Zero selected feedback dimensions without assuming an exact tensor shape."""
    feedback = feedback.clone()
    flat = feedback.reshape(-1)
    for idx in indices:
        if idx < flat.numel():
            flat[idx] = 0.0
    return feedback


def resolve_active_feedback_mode(schedule_mode, step, total_steps, switch_every=25, phase_ratio=0.5):
    """
    Resolve a high-level feedback schedule into the active branch used at this DDS step.

    Base modes:
      - only_clap
      - only_detac
      - clap_detac
      - no_feedback

    Scheduled modes:
      - alternate_clap_detac: only_clap, only_detac, only_clap, ...
      - clap_then_detac: semantic alignment first, structure preservation later
      - detac_then_clap: structure preservation first, semantic alignment later
      - cycle_clap_detac_both: only_clap -> only_detac -> clap_detac -> repeat
    """
    mode = str(schedule_mode)
    if mode in ["only_clap", "only_detac", "clap_detac", "no_feedback"]:
        return mode

    switch_every = max(1, int(switch_every))
    block_id = int(step) // switch_every

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

    raise ValueError(f"Unknown feedback schedule mode: {schedule_mode}")


def make_ablation_feedback_tensor(mode, s_edit, s_pres, previous_feedback, device, dtype):
    """
    Build the 6-D feedback vector consumed by FISC.

    Layout assumed by steermusic_utils.make_feedback_tensor:
      [edit, preserve, reference, delta_edit, delta_preserve, delta_reference]

    Active branches:
      - only_clap:  keep edit + delta_edit only
      - only_detac: keep preserve + delta_preserve only
      - clap_detac: keep edit/preserve and their deltas
      - no_feedback: all zeros
    """
    mode = str(mode)
    if mode == "no_feedback":
        return torch.zeros_like(previous_feedback, device=device, dtype=dtype)

    keep_edit = mode in ["only_clap", "clap_detac"]
    keep_pres = mode in ["only_detac", "clap_detac"]

    s_edit_value = _safe_float(s_edit) if keep_edit else 0.0
    s_pres_value = _safe_float(s_pres) if keep_pres else 0.0

    current_feedback = steermusic_utils.make_feedback_tensor(
        s_edit=s_edit_value,
        s_pres=s_pres_value,
        s_ref=0.0,
        prev_feedback=previous_feedback,
        reference_mask=0.0,
        device=device,
        dtype=dtype,
    )
    current_feedback = current_feedback.to(device=device, dtype=dtype)

    if mode == "only_clap":
        current_feedback = _zero_feedback_indices(current_feedback, [1, 2, 4, 5])
    elif mode == "only_detac":
        current_feedback = _zero_feedback_indices(current_feedback, [0, 2, 3, 5])
    elif mode == "clap_detac":
        current_feedback = _zero_feedback_indices(current_feedback, [2, 5])
    else:
        raise ValueError(f"Unknown active feedback mode: {mode}")

    return current_feedback.detach()


def best_latent_selection_score(mode, s_edit, s_pres, current_feedback, previous_feedback):
    """Use the same active branches to select best_latent for each ablation."""
    try:
        over = steermusic_utils.over_edit_penalty(current_feedback, previous_feedback)
        over_value = _safe_float(over)
    except Exception:
        over_value = 0.0

    if mode == "only_clap":
        return _safe_float(s_edit) - over_value
    if mode == "only_detac":
        return _safe_float(s_pres) - over_value
    if mode == "no_feedback":
        return -over_value
    return _safe_float(s_edit) + _safe_float(s_pres) - over_value

def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)

    print("[INFO] Running file:", os.path.abspath(__file__))
    print("[INFO] audio_path:", opt.audio_path)
    print("[INFO] prompt_ref:", opt.prompt_ref)
    print("[INFO] prompt:", opt.prompt)
    print("[INFO] fisc_ckpt:", opt.fisc_ckpt)
    print("[INFO] lambda_max:", opt.lambda_max)
    print("[INFO] requested fisc_audio_feature_dim:", opt.fisc_audio_feature_dim)
    print("[INFO] fisc_strength:", opt.fisc_strength)
    print("[INFO] edit_strength:", opt.edit_strength)
    print("[INFO] feedback_mode:", opt.feedback_mode)
    print("[INFO] feedback_switch_every:", opt.feedback_switch_every)
    print("[INFO] feedback_phase_ratio:", opt.feedback_phase_ratio)

    use_fisc = bool(opt.fisc_ckpt) and float(opt.lambda_max) > 0.0
    if use_fisc:
        print("[INFO] FISC enabled.")
    else:
        print("[INFO] FISC disabled. Baseline mode: use original target embeddings.")

    config = yaml.load(open(opt.config_yaml, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)

    guidance_model = steermusic_utils.AudioLDM2_pipe(
        opt.device,
        fp16=False,
        vram_O=False,
        t_range=[0.02, 0.98],
    )
    guidance_model.eval()
    for p in guidance_model.parameters():
        p.requires_grad = False

    log_mel_spec, _, _, _ = preprocessor.read_audio_file(filename=opt.audio_path)
    log_mel_spec = log_mel_spec.unsqueeze(0).unsqueeze(0).to(opt.device)

    with torch.no_grad():
        source_latent = guidance_model.encode_audio(log_mel_spec)
        latent = source_latent.clone().detach().requires_grad_(True)

        c_s, c_s_gen, mask_s = guidance_model.get_text_embeds(opt.prompt_ref)
        c_t, c_t_gen, mask_t = guidance_model.get_text_embeds(opt.prompt)

    fisc = None
    fisc_feature_dim = opt.fisc_audio_feature_dim

    if use_fisc:
        fisc, fisc_feature_dim = build_fisc_model(opt)
        fisc.eval()
        print("[INFO] effective FISC audio feature dim:", fisc_feature_dim)

        # warmup 创建 text-side LazyLinear 参数；audio projector 已经是显式 32000 维，不再由 LazyLinear 推断。
        feedback0 = torch.zeros(6, device=opt.device, dtype=source_latent.dtype)
        t0 = torch.tensor([guidance_model.min_step], device=opt.device, dtype=torch.long)
        with torch.no_grad():
            source_audio_feature0 = normalize_fisc_audio_feature(
                source_latent.detach(),
                target_dim=fisc_feature_dim,
            )
            _ = fisc(
                target_prompt_embeds=c_t,
                target_generated_embeds=c_t_gen,
                source_prompt_embeds=c_s,
                source_generated_embeds=c_s_gen,
                source_audio_feature=source_audio_feature0,
                feedback=feedback0,
                timestep=t0,
                reference_feature=None,
                reference_mask=0.0,
            )

        # warmup 后再加载 checkpoint
        steermusic_utils.load_fisc_checkpoint(
            fisc,
            opt.fisc_ckpt,
            strict=False,
            map_location=opt.device,
        )

        # 加载 checkpoint 后再次检测一次，确认没有发生 32000 -> 512 的错误。
        fisc_feature_dim = get_fisc_audio_feature_dim(fisc, fisc_feature_dim)
        print("[INFO] effective FISC audio feature dim after ckpt:", fisc_feature_dim)

    optim = torch.optim.SGD([latent], lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.9)

    previous_feedback = torch.zeros(6, device=opt.device, dtype=source_latent.dtype)
    best_latent = latent.detach().clone()
    best_score = -1e9

    print("[INFO] DDS editing start")
    for i in tqdm(range(opt.validation_step + 1)):
        optim.zero_grad()
        x = latent

        t = torch.randint(
            guidance_model.min_step,
            guidance_model.max_step + 1,
            [x.shape[0]],
            dtype=torch.long,
            device=opt.device,
        )
        noise = torch.randn_like(x)

        if use_fisc:
            source_audio_feature = normalize_fisc_audio_feature(
                source_latent.detach(),
                target_dim=fisc_feature_dim,
            )

            out = fisc(
                target_prompt_embeds=c_t,
                target_generated_embeds=c_t_gen,
                source_prompt_embeds=c_s,
                source_generated_embeds=c_s_gen,
                source_audio_feature=source_audio_feature,
                feedback=previous_feedback,
                timestep=t,
                reference_feature=None,
                reference_mask=0.0,
            )

            c_t_fisc = out["prompt_embeds"]
            c_t_gen_fisc = out["generated_prompt_embeds"]

            if float(opt.fisc_strength) != 1.0:
                c_t_use = c_t + float(opt.fisc_strength) * (c_t_fisc - c_t)
                c_t_gen_use = c_t_gen + float(opt.fisc_strength) * (c_t_gen_fisc - c_t_gen)
            else:
                c_t_use = c_t_fisc
                c_t_gen_use = c_t_gen_fisc

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
                    lambda_mean = lambda_prompt.float().mean().item() if lambda_prompt is not None else 0.0
                    applied_delta_ratio = (c_t_use.float() - c_t.float()).norm().item() / max(
                        c_t.float().norm().item(), 1e-6
                    )
                    print(
                        "[FISC]",
                        f"step={i}",
                        f"lambda_mean={lambda_mean:.6f}",
                        f"delta_prompt_norm={delta_prompt_norm:.6f}",
                        f"target_prompt_norm={target_prompt_norm:.6f}",
                        f"delta_ratio={delta_ratio:.6f}",
                        f"applied_delta_ratio={applied_delta_ratio:.6f}",
                    )
        else:

            c_t_use = c_t
            c_t_gen_use = c_t_gen

        # target score
        noise_pred, _, _ = guidance_model.predict_noise(
            prompt_embds=c_t_use,
            generated_prompt_embds=c_t_gen_use,
            attention_mask=mask_t,
            mel_spec=x,
            guidance_scale=opt.guidance_scale,
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
                guidance_scale=opt.guidance_scale,
                as_latent=True,
                t=t,
                noise=noise,
            )


        w = opt.weight_aug * (1 - guidance_model.alphas[t])
        grad = torch.nan_to_num(float(opt.edit_strength) * w * (noise_pred - noise_pred_src))
        loss = steermusic_utils.SpecifyGradient.apply(x, grad)
        loss.backward()
        optim.step()
        scheduler.step()

        # Build the feedback vector according to the requested ablation mode.
        # This is the part that makes only_clap vs clap_detac actually different:
        # the next FISC forward pass receives different feedback dimensions.
        with torch.no_grad():
            s_edit = steermusic_utils.noise_edit_score(noise_pred, noise_pred_src)
            s_pres = steermusic_utils.latent_preservation_score(latent, source_latent)

            active_feedback_mode = resolve_active_feedback_mode(
                schedule_mode=opt.feedback_mode,
                step=i,
                total_steps=opt.validation_step,
                switch_every=opt.feedback_switch_every,
                phase_ratio=opt.feedback_phase_ratio,
            )

            current_feedback = make_ablation_feedback_tensor(
                mode=active_feedback_mode,
                s_edit=s_edit,
                s_pres=s_pres,
                previous_feedback=previous_feedback,
                device=opt.device,
                dtype=source_latent.dtype,
            )
            score = best_latent_selection_score(
                mode=active_feedback_mode,
                s_edit=s_edit,
                s_pres=s_pres,
                current_feedback=current_feedback,
                previous_feedback=previous_feedback,
            )
            if float(score) > best_score:
                best_score = float(score)
                best_latent = latent.detach().clone()

            if i % 50 == 0:
                fb = current_feedback.detach().float().reshape(-1).cpu().tolist()
                print(
                    "[FEEDBACK]",
                    f"step={i}",
                    f"schedule={opt.feedback_mode}",
                    f"active={active_feedback_mode}",
                    f"s_edit={_safe_float(s_edit):.6f}",
                    f"s_pres_raw={_safe_float(s_pres):.6f}",
                    "vector=" + ",".join(f"{v:.6f}" for v in fb),
                    f"best_score={float(score):.6f}",
                )

            previous_feedback = current_feedback.detach()

    suffix_best = "fisc_best" if use_fisc else "baseline_best"
    suffix_last = "fisc_last" if use_fisc else "baseline_last"
    best_path = save_audio(guidance_model, best_latent, opt.audio_path, opt.output_dir, suffix_best)
    last_path = save_audio(guidance_model, latent, opt.audio_path, opt.output_dir, suffix_last)

    print("[INFO] Saved best:", best_path)
    print("[INFO] Saved last:", last_path)


if __name__ == "__main__":
    main()
