import argparse
import csv
import json
import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

import steermusic_utils as steermusic_utils
from preprocessor import Preprocessor


# =============================================================================
# 使用 ZoME-Bench 训练 FISC-SteerMusic 新增模块
# =============================================================================
# 本脚本训练改进方案中设计的网络：
#   1. SourceAudioProjector：源音频特征投影 W_s
#   2. ReferenceAudioProjector：参考音频特征投影 W_r
#   3. ReferenceSemanticAdapter A_omega：CrossAttention(Q=c_t,K=e_r,V=e_r)
#   4. PromptCompensationNetwork P_phi：2-layer Transformer Adapter
#   5. FeedbackGate G_psi：3-layer MLP + sigmoid
#
# 冻结部分：
#   AudioLDM2、VAE、U-Net、text encoder、SteerMusic DDS 主体。
#
# 重要说明：
#   ZoME-Bench 通常只公开 prompt/metadata，不直接打包 wav。
#   你需要先根据 metadata 中的 YouTube 信息下载并裁剪音频，
#   然后用 --audio_root 指向音频所在目录。
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="ZoME-Bench metadata_with_audio.parquet/csv/json/jsonl")
    parser.add_argument("--audio_root", type=str, default="", help="本地裁剪后的音频目录；当 audio_path 是相对路径时会用它拼接")
    parser.add_argument("--output_dir", type=str, default="./fisc_checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--lambda_max", type=float, default=0.05)
    parser.add_argument("--beta_pres", type=float, default=1.0)
    parser.add_argument("--rho_reg", type=float, default=0.05)
    parser.add_argument("--eta_over", type=float, default=0.2)
    parser.add_argument(
        "--semantic_reward_scale",
        type=float,
        default=100.0,
        help="Scale used by the differentiable score-level semantic reward during training.",
    )
    parser.add_argument(
        "--structural_reward_scale",
        type=float,
        default=1.0,
        help="Scale used by the differentiable score-level structural reward during training.",
    )
    parser.add_argument("--save_every", type=int, default=500)
    # 中文注释：你的 metadata_with_audio.parquet 字段通常是：
    # original_prompt / editing_prompt / editing_instruction / audio_path。
    # 因此这里默认直接使用已下载音频的 audio_path。
    parser.add_argument("--source_prompt_key", type=str, default="original_prompt")
    parser.add_argument("--target_prompt_key", type=str, default="editing_prompt")
    parser.add_argument("--audio_key", type=str, default="audio_path")
    parser.add_argument("--allow_missing_audio", action="store_true", help="允许缺失音频的行进入训练循环；默认会在训练前过滤")
    parser.add_argument("--min_audio_size", type=int, default=1024, help="音频文件最小字节数，用于过滤空/损坏文件")
    parser.add_argument("--config_yaml", type=str, default="config/autoencoder/16k_64.yaml")
    parser.add_argument(
        "--fisc_audio_feature_dim",
        type=int,
        default=32000,
        help="FISC source_audio_projector 期望的输入维度；当前 steermusic_utils.py 中通常是 32000",
    )
    return parser.parse_args()


def read_metadata(path):
    # 中文注释：读取 ZoME-Bench 的 metadata，兼容 parquet / csv / json / jsonl。
    # ZoME-Bench 在 Hugging Face 上常见格式是 .parquet，所以这里新增 parquet 支持。
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "读取 .parquet 需要 pandas 和 pyarrow，请先运行：pip install pandas pyarrow"
            ) from exc
        df = pd.read_parquet(path)
        # 中文注释：把 pandas 的 NaN 转成空字符串，避免后续字段判断出错。
        df = df.fillna("")
        return df.to_dict("records")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    if suffix in [".jsonl", ".ndjson"]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, list) else obj.get("data", [])

    raise ValueError(f"不支持的 metadata 格式: {path}")

def pick_field(row, explicit_key, candidates):
    # 中文注释：兼容不同 ZoME-Bench metadata 字段名。
    if explicit_key:
        return row.get(explicit_key, "")
    for key in candidates:
        if key in row and row[key] not in [None, ""]:
            return row[key]
    return ""


def _to_int_string(x):
    # 中文注释：把 30.0 / "30.0" 统一转成 "30"，用于拼接下载脚本生成的文件名。
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def resolve_audio_path(audio_root, audio_value, row=None, min_audio_size=1024):
    """
    中文注释：将 metadata 中的 audio_path/文件名/ytid 映射到本地音频路径。

    适配你刚下载好的 ZoME-Bench 音频：
    - 优先读取 metadata_with_audio.parquet 里的 audio_path；
    - audio_path 可以是绝对路径，也可以是相对路径；
    - 如果没有 audio_path，但有 ytid/start_s/end_s，则尝试在 audio_root 下找
      {ytid}_{start_s}_{end_s}.wav。
    """
    candidates = []

    if audio_value not in [None, ""]:
        audio_text = str(audio_value)
        p = Path(audio_text).expanduser()
        candidates.append(p)

        if audio_root and not p.is_absolute():
            root = Path(audio_root).expanduser()
            candidates.append(root / audio_text)

            stem = Path(audio_text).stem
            for ext in [".wav", ".flac", ".mp3"]:
                candidates.append(root / f"{stem}{ext}")

    if row is not None and audio_root:
        ytid = row.get("ytid", "")
        start_s = row.get("start_s", "")
        end_s = row.get("end_s", "")
        if ytid not in [None, ""] and start_s not in [None, ""] and end_s not in [None, ""]:
            fname = f"{ytid}_{_to_int_string(start_s)}_{_to_int_string(end_s)}.wav"
            candidates.append(Path(audio_root).expanduser() / fname)

    seen = set()
    for cand in candidates:
        cand = Path(cand)
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists() and cand.is_file() and cand.stat().st_size > min_audio_size:
            return str(cand)

    return ""


def prepare_training_rows(rows, opt):
    """
    中文注释：训练前先把 prompt 和 audio_path 解析好，并过滤无效样本。
    这样可以确认脚本确实在使用你下载好的 wav 文件。
    """
    prepared = []
    missing_source = 0
    missing_target = 0
    missing_audio = 0

    for row in rows:
        source_prompt = pick_field(
            row,
            opt.source_prompt_key,
            ["original_prompt", "source_prompt", "prompt_ref", "src_prompt", "source"],
        )
        target_prompt = pick_field(
            row,
            opt.target_prompt_key,
            ["editing_prompt", "target_prompt", "prompt", "tgt_prompt", "target", "edited_prompt", "editing_instruction", "instruction"],
        )
        audio_value = pick_field(
            row,
            opt.audio_key,
            ["audio_path", "path", "wav", "file", "filename", "audio"],
        )
        audio_path = resolve_audio_path(
            opt.audio_root,
            audio_value,
            row=row,
            min_audio_size=opt.min_audio_size,
        )

        if not source_prompt:
            missing_source += 1
        if not target_prompt:
            missing_target += 1
        if not audio_path:
            missing_audio += 1

        if source_prompt and target_prompt and (audio_path or opt.allow_missing_audio):
            new_row = dict(row)
            new_row["_source_prompt"] = source_prompt
            new_row["_target_prompt"] = target_prompt
            new_row["_audio_path"] = audio_path
            prepared.append(new_row)

    print("[INFO] metadata 原始行数:", len(rows))
    print("[INFO] 可用于训练的行数:", len(prepared))
    print("[INFO] 缺 source prompt 行数:", missing_source)
    print("[INFO] 缺 target prompt 行数:", missing_target)
    print("[INFO] 缺 audio_path/音频文件 行数:", missing_audio)

    if prepared:
        ex = prepared[0]
        print("[INFO] 样例 source_prompt:", str(ex["_source_prompt"])[:120])
        print("[INFO] 样例 target_prompt:", str(ex["_target_prompt"])[:120])
        print("[INFO] 样例 audio_path:", ex["_audio_path"])

    return prepared


def normalize_fisc_audio_feature(audio_feature, target_dim=32000):
    """
    Compatibility wrapper.

    The real implementation lives in steermusic_utils_fisc_v2.py so training
    and inference always use exactly the same flatten/pad/truncate/normalize
    logic for FISC audio features.
    """
    return steermusic_utils.normalize_fisc_audio_feature(
        audio_feature,
        target_dim=target_dim,
        detach=True,
    )


def build_model(device, lambda_max):
    guidance_model = steermusic_utils.AudioLDM2_pipe(device, fp16=False, vram_O=False, t_range=[0.02, 0.98])
    guidance_model.eval()
    for p in guidance_model.parameters():
        p.requires_grad = False

    fisc = steermusic_utils.FISCModel(lambda_max=lambda_max).to(device)
    fisc.train()
    return guidance_model, fisc


def warmup_fisc(fisc, guidance_model, source_prompt, target_prompt, source_latent, device, feature_dim=32000):
    # 中文注释：LazyLinear 需要先跑一次 forward 创建参数，然后才能创建 optimizer。
    with torch.no_grad():
        c_s, c_s_gen, _ = guidance_model.get_text_embeds(source_prompt)
        c_t, c_t_gen, _ = guidance_model.get_text_embeds(target_prompt)
    # Only used to initialize lazy FIS2C layers before creating the optimizer.
    # Actual training feedback is computed from the uncompensated base state.
    feedback0 = torch.zeros(6, device=device, dtype=source_latent.dtype)
    t0 = torch.tensor([guidance_model.min_step], device=device, dtype=torch.long)
    source_audio_feature = steermusic_utils.normalize_fisc_audio_feature(source_latent.detach(), target_dim=feature_dim)
    _ = fisc(c_t, c_t_gen, c_s, c_s_gen, source_audio_feature, feedback0, t0)



def _score_level_semantic_reward(edit_direction, scale=100.0, eps=1e-6):
    """
    Differentiable train-time semantic reward defined on the sampled DDS state.

    This is intentionally kept in the autograd graph for the compensated branch.
    The base branch is detached separately. It is a score-level training surrogate;
    inference-time CLAP feedback remains unchanged.
    """
    flat = edit_direction.flatten(1)
    strength = flat.norm(dim=1) / max(float(scale), eps)
    return torch.tanh(strength).mean()


def _score_level_structural_reward(edit_direction, source_latent, scale=1.0, eps=1e-6):
    """
    Differentiable train-time structural reward on the sampled DDS state.

    Larger DDS perturbations relative to the source latent receive lower
    preservation reward. The function is smooth so gradients can reach FIS2C.
    """
    dir_norm = edit_direction.flatten(1).norm(dim=1)
    src_norm = source_latent.detach().flatten(1).norm(dim=1).clamp_min(eps)
    relative_change = dir_norm / src_norm
    return torch.exp(-float(scale) * relative_change.pow(2)).mean()


def _feedback_improvement_loss(
    r_sem_base,
    r_str_base,
    r_ref_base,
    r_sem_fis2c,
    r_str_fis2c,
    r_ref_fis2c,
    reference_mask=0.0,
):
    """
    L_over from Sec. 4.3:
      -(r_sem^FIS2C-r_sem^base)
      -(r_str^FIS2C-r_str^base)
      -m_r(r_ref^FIS2C-r_ref^base)

    Base rewards are stop-gradient references; compensated rewards keep gradients.
    """
    mr = torch.as_tensor(
        float(reference_mask),
        device=r_sem_fis2c.device,
        dtype=r_sem_fis2c.dtype,
    )
    return (
        -(r_sem_fis2c - r_sem_base.detach())
        -(r_str_fis2c - r_str_base.detach())
        -mr * (r_ref_fis2c - r_ref_base.detach())
    )


def train_one_sample(row, opt, guidance_model, fisc, preprocessor, optimizer, global_step):
    device = opt.device

    source_prompt = row.get("_source_prompt") or pick_field(
        row, opt.source_prompt_key, ["original_prompt", "source_prompt", "prompt_ref", "src_prompt", "source"]
    )
    target_prompt = row.get("_target_prompt") or pick_field(
        row, opt.target_prompt_key, ["editing_prompt", "target_prompt", "prompt", "tgt_prompt", "target", "edited_prompt", "editing_instruction", "instruction"]
    )
    audio_path = row.get("_audio_path", "")
    if not audio_path:
        audio_value = pick_field(row, opt.audio_key, ["audio_path", "path", "wav", "file", "filename", "audio"])
        audio_path = resolve_audio_path(opt.audio_root, audio_value, row=row, min_audio_size=opt.min_audio_size)

    if not source_prompt or not target_prompt or not audio_path:
        return None

    log_mel_spec, _, _, _ = preprocessor.read_audio_file(filename=audio_path)
    log_mel_spec = log_mel_spec.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Frozen source audio / text encoders.
        source_latent = guidance_model.encode_audio(log_mel_spec)
        c_s, c_s_gen, mask_s = guidance_model.get_text_embeds(source_prompt)
        c_t, c_t_gen, mask_t = guidance_model.get_text_embeds(target_prompt)

    # One sampled DDS state, shared by base and compensated branches.
    t = torch.randint(
        guidance_model.min_step,
        guidance_model.max_step + 1,
        [1],
        dtype=torch.long,
        device=device,
    )
    noise = torch.randn_like(source_latent)

    source_audio_feature = steermusic_utils.normalize_fisc_audio_feature(
        source_latent.detach(),
        target_dim=opt.fisc_audio_feature_dim,
    )

    # ------------------------------------------------------------------
    # 1) Frozen base/source branches -> non-zero F_base
    # ------------------------------------------------------------------
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
            disable_grad=True,
        )

        noise_pred_tgt_base, _, _ = guidance_model.predict_noise(
            prompt_embds=c_t,
            generated_prompt_embds=c_t_gen,
            attention_mask=mask_t,
            mel_spec=source_latent,
            guidance_scale=opt.guidance_scale,
            as_latent=True,
            t=t,
            noise=noise,
            disable_grad=True,
        )

        d_base = (noise_pred_tgt_base - noise_pred_src).detach()

        r_sem_base = _score_level_semantic_reward(
            d_base,
            scale=opt.semantic_reward_scale,
        ).detach()
        r_str_base = _score_level_structural_reward(
            d_base,
            source_latent,
            scale=opt.structural_reward_scale,
        ).detach()
        r_ref_base = torch.zeros([], device=device, dtype=r_sem_base.dtype)

        # F_base = [r_sem, r_str, r_ref, delta_sem, delta_str, delta_ref].
        # There is no previous feedback round during single-step training,
        # therefore the delta components are initialized to zero.
        F_base = steermusic_utils.make_feedback_tensor(
            s_edit=float(r_sem_base.cpu()),
            s_pres=float(r_str_base.cpu()),
            s_ref=0.0,
            prev_feedback=None,
            reference_mask=0.0,
            device=device,
            dtype=source_latent.dtype,
        ).detach()

    # ------------------------------------------------------------------
    # 2) Feedback-conditioned compensation
    # ------------------------------------------------------------------
    out = fisc(
        target_prompt_embeds=c_t,
        target_generated_embeds=c_t_gen,
        source_prompt_embeds=c_s,
        source_generated_embeds=c_s_gen,
        source_audio_feature=source_audio_feature,
        feedback=F_base,
        timestep=t,
        reference_feature=None,
        reference_mask=0.0,
    )

    # Only this compensated target branch retains gradients.
    noise_pred_tgt_fis2c, _, _ = guidance_model.predict_noise(
        prompt_embds=out["prompt_embeds"],
        generated_prompt_embds=out["generated_prompt_embeds"],
        attention_mask=mask_t,
        mel_spec=source_latent,
        guidance_scale=opt.guidance_scale,
        as_latent=True,
        t=t,
        noise=noise,
        disable_grad=False,
    )

    eps = 1e-6
    d_fis2c = noise_pred_tgt_fis2c - noise_pred_src.detach()

    d_base_flat = d_base.flatten(1)
    d_fis2c_flat = d_fis2c.flatten(1)

    # ------------------------------------------------------------------
    # 3) DDS-prior regularization
    # ------------------------------------------------------------------
    l_edit_dir = 1.0 - torch.nn.functional.cosine_similarity(
        d_fis2c_flat,
        d_base_flat,
        dim=1,
        eps=eps,
    ).mean()

    base_norm = d_base_flat.norm(dim=1).detach()
    fis2c_norm = d_fis2c_flat.norm(dim=1)
    mag_ratio = fis2c_norm / (base_norm + eps)

    # Penalize only compensation that is >20% stronger than the base DDS direction.
    l_edit_mag = torch.relu(mag_ratio - 1.20).pow(2).mean()

    delta_prompt = out.get("delta_prompt", None)
    delta_generated = out.get("delta_generated", None)
    l_delta_prompt = (
        delta_prompt.float().pow(2).mean()
        if delta_prompt is not None
        else torch.zeros([], device=device)
    )
    l_delta_generated = (
        delta_generated.float().pow(2).mean()
        if delta_generated is not None
        else torch.zeros([], device=device)
    )
    l_delta = l_delta_prompt + l_delta_generated

    lambda_prompt = out.get("lambda_prompt", None)
    lambda_generated = out.get("lambda_generated", None)
    l_lambda_prompt = (
        lambda_prompt.float().pow(2).mean()
        if lambda_prompt is not None
        else torch.zeros([], device=device)
    )
    l_lambda_generated = (
        lambda_generated.float().pow(2).mean()
        if lambda_generated is not None
        else torch.zeros([], device=device)
    )
    l_lambda = l_lambda_prompt + l_lambda_generated

    # ------------------------------------------------------------------
    # 4) F_FIS2C and feedback-improvement objective
    # ------------------------------------------------------------------
    # IMPORTANT: these compensated rewards stay in the autograd graph.
    r_sem_fis2c = _score_level_semantic_reward(
        d_fis2c,
        scale=opt.semantic_reward_scale,
    )
    r_str_fis2c = _score_level_structural_reward(
        d_fis2c,
        source_latent,
        scale=opt.structural_reward_scale,
    )
    r_ref_fis2c = torch.zeros([], device=device, dtype=r_sem_fis2c.dtype)

    l_over = _feedback_improvement_loss(
        r_sem_base=r_sem_base,
        r_str_base=r_str_base,
        r_ref_base=r_ref_base,
        r_sem_fis2c=r_sem_fis2c,
        r_str_fis2c=r_str_fis2c,
        r_ref_fis2c=r_ref_fis2c,
        reference_mask=0.0,
    )

    # Detached F_FIS2C is kept only for logging / inspection.
    F_fis2c = steermusic_utils.make_feedback_tensor(
        s_edit=float(r_sem_fis2c.detach().cpu()),
        s_pres=float(r_str_fis2c.detach().cpu()),
        s_ref=0.0,
        prev_feedback=F_base,
        reference_mask=0.0,
        device=device,
        dtype=source_latent.dtype,
    ).detach()

    loss = (
        l_edit_dir
        + 0.20 * l_edit_mag
        + opt.rho_reg * l_delta
        + 0.01 * l_lambda
        + opt.eta_over * l_over
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(fisc.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "r_sem_base": float(r_sem_base.detach().cpu()),
        "r_str_base": float(r_str_base.detach().cpu()),
        "r_sem_fis2c": float(r_sem_fis2c.detach().cpu()),
        "r_str_fis2c": float(r_str_fis2c.detach().cpu()),
        "l_over": float(l_over.detach().cpu()),
        "l_edit_dir": float(l_edit_dir.detach().cpu()),
        "l_edit_mag": float(l_edit_mag.detach().cpu()),
        "l_delta": float(l_delta.detach().cpu()),
        "l_lambda": float(l_lambda.detach().cpu()),
        "mag_ratio": float(mag_ratio.mean().detach().cpu()),
        "F_base": [float(x) for x in F_base.detach().cpu().flatten()],
        "F_fis2c": [float(x) for x in F_fis2c.detach().cpu().flatten()],
        "global_step": global_step,
    }


def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)

    rows = read_metadata(opt.metadata)
    rows = prepare_training_rows(rows, opt)
    if opt.max_samples > 0:
        rows = rows[: opt.max_samples]
        print("[INFO] 应用 --max_samples 后训练行数:", len(rows))

    if not rows:
        raise RuntimeError(
            "没有找到可用样本。请确认 --metadata 指向 metadata_with_audio.parquet，"
            "并且 audio_path 文件真实存在。"
        )

    config = yaml.load(open(opt.config_yaml, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)

    guidance_model, fisc = build_model(opt.device, opt.lambda_max)

    first = None
    for row in rows:
        audio_path = row.get("_audio_path", "")
        sp = row.get("_source_prompt", "")
        tp = row.get("_target_prompt", "")
        if audio_path and sp and tp:
            first = (audio_path, sp, tp)
            break

    if first is None:
        raise RuntimeError("没有找到可用样本，请检查 metadata 字段名、audio_path 和 --audio_root。")

    log_mel_spec, _, _, _ = preprocessor.read_audio_file(filename=first[0])
    log_mel_spec = log_mel_spec.unsqueeze(0).unsqueeze(0).to(opt.device)
    with torch.no_grad():
        source_latent = guidance_model.encode_audio(log_mel_spec)
    warmup_fisc(fisc, guidance_model, first[1], first[2], source_latent, opt.device, feature_dim=opt.fisc_audio_feature_dim)

    # 中文注释：optimizer 只训练 FISC 新增模块参数。
    optimizer = torch.optim.AdamW(fisc.parameters(), lr=opt.lr, weight_decay=1e-4)

    global_step = 0
    for epoch in range(opt.epochs):
        pbar = tqdm(rows, desc=f"epoch {epoch+1}/{opt.epochs}")
        for row in pbar:
            metrics = train_one_sample(row, opt, guidance_model, fisc, preprocessor, optimizer, global_step)
            if metrics is None:
                continue
            global_step += 1
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}", r_sem=f"{metrics.get('r_sem_fis2c', 0.0):.3f}", r_str=f"{metrics.get('r_str_fis2c', 0.0):.3f}", l_over=f"{metrics.get('l_over', 0.0):.3f}")

            if global_step % opt.save_every == 0:
                ckpt_path = os.path.join(opt.output_dir, f"fisc_step_{global_step}.pt")
                steermusic_utils.save_fisc_checkpoint(fisc, ckpt_path, extra={"global_step": global_step, "epoch": epoch})
                print(f"[INFO] saved {ckpt_path}")

    final_path = os.path.join(opt.output_dir, "fisc_final.pt")
    steermusic_utils.save_fisc_checkpoint(fisc, final_path, extra={"global_step": global_step, "epochs": opt.epochs})
    print(f"[INFO] training finished, saved {final_path}")


if __name__ == "__main__":
    main()
