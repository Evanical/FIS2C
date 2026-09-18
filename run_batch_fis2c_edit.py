#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch_edit_feedback_schedule.py

Batch runner for FISC-SteerMusic ordinary editing with scheduled feedback branches.
Writes batch_status.jsonl compatible with eval/test.py.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--audio_root", default="")
    p.add_argument("--output_root", required=True)
    p.add_argument("--steermusic_script", default="./fis2c_edit.py")
    p.add_argument("--fisc_ckpt", required=True)
    p.add_argument("--python_bin", default=sys.executable)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--validation_step", type=int, default=500)
    p.add_argument("--guidance_scale", type=float, default=30.0)
    p.add_argument("--weight_aug", type=float, default=2.0)
    p.add_argument("--lambda_max", type=float, default=0.03)
    p.add_argument("--fisc_audio_feature_dim", type=int, default=32000)
    p.add_argument("--fisc_strength", type=float, default=1.3)
    p.add_argument("--edit_strength", type=float, default=1.3)
    p.add_argument(
        "--feedback_mode",
        default="alternate_clap_detac",
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
    p.add_argument("--feedback_switch_every", type=int, default=25)
    p.add_argument("--feedback_phase_ratio", type=float, default=0.5)
    p.add_argument("--config_yaml", default="config/autoencoder/16k_64.yaml")

    p.add_argument("--source_prompt_key", default="original_prompt")
    p.add_argument("--target_prompt_key", default="editing_prompt")
    p.add_argument("--audio_key", default="audio_path")
    p.add_argument("--ytid_key", default="ytid")
    p.add_argument("--index_key", default="original_index")

    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--print_subprocess", action="store_true")
    return p.parse_args()


def read_metadata(path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p).fillna("")
    if suf == ".csv":
        return pd.read_csv(p).fillna("")
    if suf in [".jsonl", ".ndjson"]:
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        return pd.DataFrame(rows).fillna("")
    if suf == ".json":
        try:
            return pd.read_json(p).fillna("")
        except ValueError as e:
            if "Trailing data" not in str(e) and "Expected object or value" not in str(e):
                raise
            rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            return pd.DataFrame(rows).fillna("")
    raise ValueError(f"Unsupported metadata format: {p}")


def is_empty(x):
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def sanitize_name(s, max_len=80):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return (s or "sample")[:max_len]


def to_int_string(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def get_original_index(row, args, fallback):
    for key in [args.index_key, "original_index", "index"]:
        if key in row and not is_empty(row[key]):
            try:
                return int(float(row[key]))
            except Exception:
                pass
    return int(fallback)


def resolve_audio_path(row, args, metadata_dir):
    root = Path(args.audio_root).expanduser() if args.audio_root else None
    candidates = []

    if args.audio_key in row and not is_empty(row[args.audio_key]):
        raw = str(row[args.audio_key]).strip()
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            if root is not None:
                candidates.append(root / p)
                candidates.append(root / p.name)
            candidates.append(metadata_dir / p)
            candidates.append(metadata_dir / "audio" / p)
            candidates.append(metadata_dir / "audio" / p.name)

    ytid = str(row.get(args.ytid_key, "")).strip()
    start_s = str(row.get("start_s", "")).strip()
    end_s = str(row.get("end_s", "")).strip()
    if root is not None and ytid:
        if start_s and end_s:
            candidates.append(root / f"{ytid}_{to_int_string(start_s)}_{to_int_string(end_s)}.wav")
        candidates.extend(sorted(root.glob(f"{ytid}_*.wav")))

    seen = set()
    for c in candidates:
        c = Path(c)
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return str(c.resolve())
    return ""


def output_exists(output_dir):
    p = Path(output_dir)
    if not p.exists():
        return False
    return any(p.glob("*_fisc_best.wav")) or any(p.glob("*_fisc_last.wav"))


def first_match(folder, patterns):
    for pat in patterns:
        matches = sorted(Path(folder).glob(pat))
        if matches:
            return str(matches[0].resolve())
    return ""


def run_subprocess(cmd, log_path, print_subprocess):
    t0 = time.perf_counter()
    if print_subprocess:
        ret = subprocess.run(cmd)
    else:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("[CMD] " + " ".join(cmd) + "\n\n")
            f.flush()
            ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return ret.returncode, time.perf_counter() - t0


def main():
    args = parse_args()
    metadata_path = Path(args.metadata).resolve()
    metadata_dir = metadata_path.parent
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df = read_metadata(metadata_path)
    print("[INFO] metadata:", metadata_path)
    print("[INFO] shape:", df.shape)
    print("[INFO] columns:", list(df.columns))
    print("[INFO] output_root:", output_root)
    print("[INFO] feedback_mode:", args.feedback_mode)

    rows = df.iloc[int(args.start):].copy()
    if args.limit is not None and int(args.limit) >= 0:
        rows = rows.head(int(args.limit)).copy()

    status_path = output_root / "batch_status.jsonl"
    with status_path.open("w" if args.overwrite else "a", encoding="utf-8") as sf:
        for n, (_, row) in enumerate(rows.iterrows(), start=1):
            rd = row.to_dict()
            idx = get_original_index(rd, args, fallback=int(args.start) + n - 1)
            audio_path = resolve_audio_path(rd, args, metadata_dir)
            source_prompt = str(rd.get(args.source_prompt_key, "")).strip()
            target_prompt = str(rd.get(args.target_prompt_key, "")).strip()

            sample_name = sanitize_name(Path(audio_path).stem if audio_path else rd.get(args.ytid_key, f"row{idx}"))
            output_dir = output_root / f"{idx:05d}_{sample_name}"
            output_dir.mkdir(parents=True, exist_ok=True)

            rec = {
                "index": idx,
                "sample_name": sample_name,
                "audio_path": audio_path,
                "source_audio": audio_path,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "output_dir": str(output_dir),
                "feedback_mode": args.feedback_mode,
                "feedback_switch_every": args.feedback_switch_every,
                "feedback_phase_ratio": args.feedback_phase_ratio,
                "validation_step": args.validation_step,
                "status": "pending",
            }

            if not audio_path:
                rec["status"] = "missing_audio"
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sf.flush()
                continue
            if not source_prompt:
                rec["status"] = "missing_source_prompt"
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sf.flush()
                continue
            if not target_prompt:
                rec["status"] = "missing_target_prompt"
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sf.flush()
                continue

            if output_exists(output_dir) and not args.overwrite:
                rec["status"] = "skipped_existing"
                rec["best_path"] = first_match(output_dir, ["*_fisc_best.wav", "*best*.wav"])
                rec["last_path"] = first_match(output_dir, ["*_fisc_last.wav", "*last*.wav"])
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sf.flush()
                print("[SKIP]", idx, sample_name)
                continue

            cmd = [
                args.python_bin,
                args.fis2c_script,
                "--audio_path", audio_path,
                "--prompt_ref", source_prompt,
                "--prompt", target_prompt,
                "--output_dir", str(output_dir),
                "--validation_step", str(args.validation_step),
                "--guidance_scale", str(args.guidance_scale),
                "--weight_aug", str(args.weight_aug),
                "--device", args.device,
                "--fisc_ckpt", args.fisc_ckpt,
                "--lambda_max", str(args.lambda_max),
                "--fisc_audio_feature_dim", str(args.fisc_audio_feature_dim),
                "--config_yaml", args.config_yaml,
                "--fisc_strength", str(args.fisc_strength),
                "--edit_strength", str(args.edit_strength),
                "--feedback_mode", args.feedback_mode,
                "--feedback_switch_every", str(args.feedback_switch_every),
                "--feedback_phase_ratio", str(args.feedback_phase_ratio),
            ]
            rec["cmd"] = json.dumps(cmd, ensure_ascii=False)

            if args.dry_run:
                rec["status"] = "dry_run"
                print("[DRY_RUN]", " ".join(cmd))
                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sf.flush()
                continue

            print("=" * 100)
            print(f"[RUN] {n}/{len(rows)} idx={idx} mode={args.feedback_mode}")
            print("[OUT]", output_dir)

            log_path = output_dir / "run.log"
            rc, elapsed = run_subprocess(cmd, log_path, args.print_subprocess)

            rec["returncode"] = int(rc)
            rec["elapsed_seconds"] = float(elapsed)
            rec["best_path"] = first_match(output_dir, ["*_fisc_best.wav", "*best*.wav"])
            rec["last_path"] = first_match(output_dir, ["*_fisc_last.wav", "*last*.wav"])
            rec["log_path"] = str(log_path)
            rec["status"] = "ok" if rc == 0 and rec["best_path"] else "failed"

            sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sf.flush()
            print(f"[DONE] {rec['status']} elapsed={elapsed:.2f}s best={rec['best_path']}")

    print("[INFO] status:", status_path)


if __name__ == "__main__":
    main()
