#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--audio_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--personalized_ckpt", type=str, required=True)
    parser.add_argument("--fisc_ckpt", type=str, required=True)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--steermusic_personalized_script", type=str, default="./SteerMusic_personalized_ablation_feedback_schedule.py")

    parser.add_argument("--validation_step", type=int, default=500)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--weight_aug", type=float, default=2.0)
    parser.add_argument("--fisc_strength", type=float, default=1.5)
    parser.add_argument("--edit_strength", type=float, default=1.5)
    parser.add_argument("--lambda_max", type=float, default=0.05)
    parser.add_argument("--fisc_audio_feature_dim", type=int, default=32000)
    parser.add_argument("--feedback_every", type=int, default=25)
    parser.add_argument("--feedback_eval_script", type=str, default="./test.py")
    parser.add_argument("--feedback_device", type=str, default="cpu")
    parser.add_argument("--clap_checkpoint_dir", type=str, default="clap/pretrained")
    parser.add_argument("--clap_checkpoint_name", type=str, default="music_audioset_epoch_15_esc_90.14.pt")
    parser.add_argument("--detac_model_name", type=str, default="m-a-p/MERT-v1-95M")
    parser.add_argument("--detac_layer", type=int, default=-1)
    parser.add_argument("--detac_quantile", type=float, default=0.1)
    parser.add_argument("--detac_cache_dir", type=str, default="./eval/.detac_mert_cache")
    parser.add_argument("--detac_max_audio_seconds", type=float, default=None)
    parser.add_argument("--detac_weight", type=float, default=1.0)
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
    )
    parser.add_argument("--feedback_switch_every_rounds", type=int, default=1)
    parser.add_argument("--feedback_phase_ratio", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--config_yaml", type=str, default="config/autoencoder/16k_64.yaml")
    parser.add_argument("--audioldm2_path", type=str, default="", help="Local AudioLDM2 directory. If set, subprocess loads AudioLDM2 offline from this path.")

    parser.add_argument("--source_prompt_key", type=str, default="original_prompt")
    parser.add_argument("--target_prompt_key", type=str, default="editing_prompt")
    parser.add_argument("--audio_key", type=str, default="audio_path")
    parser.add_argument("--concept_key", type=str, default="concept")
    parser.add_argument("--default_concept", type=str, default="music")

    parser.add_argument("--ref_audio_key", type=str, default="")
    parser.add_argument("--ref_audio_path", type=str, default="")

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for subprocess.")

    return parser.parse_args()


def read_json_lines(path):
    """Read JSONL/NDJSON: one JSON object per line."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return pd.DataFrame(rows).fillna("")


def read_metadata(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path).fillna("")

    if suffix == ".csv":
        return pd.read_csv(path).fillna("")

    if suffix in [".jsonl", ".ndjson"]:
        return read_json_lines(path)

    if suffix == ".json":
        # First try normal JSON, e.g. [{...}, {...}] or {"key": ...}.
        # If pandas raises "Trailing data", the file is usually JSONL/NDJSON
        # even though its suffix is .json, so fall back to line-by-line JSON.
        try:
            return pd.read_json(path).fillna("")
        except ValueError as e:
            msg = str(e).lower()
            if "trailing data" in msg or "expected object or value" in msg:
                print(f"[WARN] {path} looks like JSONL/NDJSON; reading it line by line.")
                return read_json_lines(path)
            raise

    raise ValueError(f"Unsupported metadata format: {path}")


def to_int_string(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def resolve_audio_path(row, audio_root, audio_key="audio_path"):
    candidates = []
    root = Path(audio_root).expanduser()

    if audio_key in row and str(row.get(audio_key, "")).strip():
        raw = str(row[audio_key]).strip()
        p = Path(raw).expanduser()
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(root / raw)
            candidates.append(root / Path(raw).name)

    ytid = str(row.get("ytid", "")).strip()
    start_s = str(row.get("start_s", "")).strip()
    end_s = str(row.get("end_s", "")).strip()
    if ytid and start_s and end_s:
        candidates.append(root / f"{ytid}_{to_int_string(start_s)}_{to_int_string(end_s)}.wav")

    seen = set()
    for p in candidates:
        p = Path(p)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            return str(p.resolve())

    return ""


def get_value(row, key, default=""):
    if key and key in row and str(row.get(key, "")).strip():
        return str(row[key]).strip()
    return default


def write_status(f, record):
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


def main():
    args = parse_args()

    output_root = Path(args.output_root) / str(args.feedback_mode)
    output_root.mkdir(parents=True, exist_ok=True)

    df = read_metadata(args.metadata)
    if args.start_index > 0:
        df = df.iloc[args.start_index:].copy()
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    status_path = output_root / "batch_status.jsonl"

    print("[INFO] metadata:", args.metadata)
    print("[INFO] rows to run:", len(df))
    print("[INFO] output_root:", output_root.resolve())
    print("[INFO] status_path:", status_path.resolve())
    print("[INFO] script:", args.steermusic_personalized_script)
    print("[INFO] feedback_mode:", args.feedback_mode)
    print("[INFO] feedback_switch_every_rounds:", args.feedback_switch_every_rounds)
    print("[INFO] feedback_phase_ratio:", args.feedback_phase_ratio)

    env = os.environ.copy()
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["DIFFUSERS_OFFLINE"] = "1"
    if args.audioldm2_path:
        env["AUDIOLDM2_LOCAL_PATH"] = str(Path(args.audioldm2_path).expanduser().resolve())
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["DIFFUSERS_OFFLINE"] = "1"

    with status_path.open("a", encoding="utf-8") as f_status:
        for local_i, (_, row_obj) in enumerate(tqdm(df.iterrows(), total=len(df))):
            row = row_obj.to_dict()
            original_index = int(row.get("original_index", args.start_index + local_i)) if str(row.get("original_index", "")).strip() else args.start_index + local_i

            source_prompt = get_value(row, args.source_prompt_key)
            target_prompt = get_value(row, args.target_prompt_key)
            concept = get_value(row, args.concept_key, args.default_concept)

            audio_path = resolve_audio_path(row, args.audio_root, args.audio_key)

            if not source_prompt:
                write_status(f_status, {"index": original_index, "status": "missing_source_prompt"})
                continue

            if not audio_path:
                write_status(f_status, {"index": original_index, "status": "missing_audio", "row_audio_value": row.get(args.audio_key, "")})
                continue

            sample_name = f"{original_index:05d}_{Path(audio_path).stem}"
            output_dir = output_root / sample_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if not args.overwrite and any(output_dir.glob("*.wav")):
                write_status(f_status, {
                    "index": original_index,
                    "status": "skipped_existing",
                    "audio_path": audio_path,
                    "output_dir": str(output_dir),
                })
                continue

            cmd = [
                args.python_bin, args.steermusic_personalized_script,
                "--audio_path", audio_path,
                "--prompt_ref", source_prompt,
                "--concept", concept,
                "--personalized_ckpt", args.personalized_ckpt,
                "--output_dir", str(output_dir),
                "--guidance_scale", str(args.guidance_scale),
                "--weight_aug", str(args.weight_aug),
                "--fisc_strength", str(args.fisc_strength),
                "--edit_strength", str(args.edit_strength),
                "--validation_step", str(args.validation_step),
                "--device", args.device,
                "--lambda_max", str(args.lambda_max),
                "--fisc_audio_feature_dim", str(args.fisc_audio_feature_dim),
                "--config_yaml", args.config_yaml,
                "--audioldm2_path", args.audioldm2_path,
                "--feedback_every", str(args.feedback_every),
                "--feedback_eval_script", args.feedback_eval_script,
                "--clap_checkpoint_dir", args.clap_checkpoint_dir,
                "--clap_checkpoint_name", args.clap_checkpoint_name,
                "--detac_model_name", args.detac_model_name,
                "--detac_layer", str(args.detac_layer),
                "--detac_quantile", str(args.detac_quantile),
                "--detac_cache_dir", args.detac_cache_dir,
                "--detac_weight", str(args.detac_weight),
                "--feedback_mode", args.feedback_mode,
                "--feedback_switch_every_rounds", str(args.feedback_switch_every_rounds),
                "--feedback_phase_ratio", str(args.feedback_phase_ratio),
            ]

            if args.detac_max_audio_seconds is not None:
                cmd.extend([
                    "--detac_max_audio_seconds",
                    str(args.detac_max_audio_seconds),
                ])

            if args.fisc_ckpt:
                cmd.extend(["--fisc_ckpt", args.fisc_ckpt])

            ref_audio = args.ref_audio_path
            if args.ref_audio_key:
                ref_audio = get_value(row, args.ref_audio_key, ref_audio)

            if ref_audio:
                cmd.extend(["--ref_audio_path", ref_audio])

            ret = subprocess.run(cmd, text=True, stdout=None, stderr=None, env=env)

            record = {
                "index": original_index,
                "sample_name": sample_name,
                "status": "ok" if ret.returncode == 0 else "failed",
                "returncode": ret.returncode,
                "audio_path": audio_path,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "concept": concept,
                "guidance_scale": args.guidance_scale,
                "weight_aug": args.weight_aug,
                "fisc_strength": args.fisc_strength,
                "edit_strength": args.edit_strength,
                "ablation_mode": args.feedback_mode,
                "feedback_mode": args.feedback_mode,
                "feedback_switch_every_rounds": args.feedback_switch_every_rounds,
                "feedback_phase_ratio": args.feedback_phase_ratio,
                "output_dir": str(output_dir),
                "cmd": cmd,
                "stdout_tail": ret.stdout[-5000:] if ret.stdout else "",
                "stderr_tail": ret.stderr[-12000:] if ret.stderr else "",
            }
            write_status(f_status, record)


if __name__ == "__main__":
    main()
