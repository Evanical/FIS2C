#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deTAC.py

DTW-based Temporal Alignment Consistency for music editing.

Formula:
    f(x) = [f_1, ..., f_T]              # frame-level MERT features
    pi* = DTW(f(x_src), f(x_edit))      # DTW alignment path
    deTAC = mean_{(i,j) in pi*} cos(f_i_src, f_j_edit)
    deTAC_q = quantile_q({cos(f_i_src, f_j_edit)}_{(i,j) in pi*})

Higher is better for both deTAC and deTAC_q.

Recommended location:
    /home/ps/lmy/Plus_steermusic_main/eval/deTAC.py

Example:
    cd /home/ps/lmy/Plus_steermusic_main
    python eval/deTAC.py \
      --manifest ./SteerMusic_fisc_zome_output/generation_manifest.csv \
      --out_csv ./SteerMusic_fisc_zome_output/eval_deTAC.csv \
      --out_summary ./SteerMusic_fisc_zome_output/eval_deTAC_summary.csv \
      --device cuda:0 \
      --quantile 0.1

Quick test:
    python eval/deTAC.py \
      --manifest ./SteerMusic_fisc_zome_output/generation_manifest.csv \
      --limit 5 \
      --device cuda:0
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def cosine_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def exact_dtw_from_cosine(src_feat: np.ndarray, edit_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Exact DTW with local cost = 1 - cosine.

    Inputs should already be L2-normalized:
        src_feat:  [T, D]
        edit_feat: [U, D]

    Returns:
        path:      [L, 2] integer array of (i, j)
        path_sims: [L] cosine values along path
        total_cost: DTW accumulated cost at endpoint
    """
    src_feat = np.asarray(src_feat, dtype=np.float32)
    edit_feat = np.asarray(edit_feat, dtype=np.float32)

    if src_feat.ndim != 2 or edit_feat.ndim != 2:
        raise ValueError(f"features must be 2D, got {src_feat.shape} and {edit_feat.shape}")
    if src_feat.shape[0] == 0 or edit_feat.shape[0] == 0:
        raise ValueError(f"empty feature sequence: {src_feat.shape}, {edit_feat.shape}")
    if src_feat.shape[1] != edit_feat.shape[1]:
        raise ValueError(f"feature dim mismatch: {src_feat.shape[1]} vs {edit_feat.shape[1]}")

    sim = np.matmul(src_feat, edit_feat.T)
    sim = np.clip(sim, -1.0, 1.0)
    cost = 1.0 - sim

    T, U = cost.shape
    acc = np.full((T + 1, U + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0

    # 0 = diagonal, 1 = up, 2 = left
    back = np.zeros((T, U), dtype=np.uint8)

    for i in range(1, T + 1):
        row_cost = cost[i - 1]
        for j in range(1, U + 1):
            diag = acc[i - 1, j - 1]
            up = acc[i - 1, j]
            left = acc[i, j - 1]

            if diag <= up and diag <= left:
                acc[i, j] = row_cost[j - 1] + diag
                back[i - 1, j - 1] = 0
            elif up <= left:
                acc[i, j] = row_cost[j - 1] + up
                back[i - 1, j - 1] = 1
            else:
                acc[i, j] = row_cost[j - 1] + left
                back[i - 1, j - 1] = 2

    i, j = T - 1, U - 1
    path = []

    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break

        step = back[i, j]
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

        if i < 0:
            i = 0
        if j < 0:
            j = 0

    path.reverse()
    path = np.asarray(path, dtype=np.int64)
    path_sims = sim[path[:, 0], path[:, 1]].astype(np.float32)
    return path, path_sims, float(acc[T, U])


class MERTFeatureExtractor:
    """MERT frame-level feature extractor with optional .npy caching."""

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        device: str = "cuda:0",
        layer: int = -1,
        cache_dir: Optional[str] = "./eval/.detac_mert_cache",
        max_audio_seconds: Optional[float] = None,
    ):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.torch = torch
        self.model_name = model_name
        self.layer = layer
        self.max_audio_seconds = max_audio_seconds
        self.device = torch.device(device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache: Dict[str, np.ndarray] = {}

        print(f"[INFO] loading MERT model: {model_name}")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_hidden_states=True,
        ).to(self.device)
        self.model.eval()

        self.sample_rate = int(getattr(self.processor, "sampling_rate", 24000) or 24000)
        print(f"[INFO] MERT sample_rate: {self.sample_rate}")
        print(f"[INFO] MERT device: {self.device}")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        st = audio_path.stat()
        raw = json.dumps(
            {
                "path": str(audio_path.resolve()),
                "mtime": st.st_mtime,
                "size": st.st_size,
                "model": self.model_name,
                "layer": self.layer,
                "sr": self.sample_rate,
                "max_audio_seconds": self.max_audio_seconds,
            },
            sort_keys=True,
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def extract(self, audio_path: str) -> np.ndarray:
        import librosa

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"audio not found: {audio_path}")

        key = self._cache_key(audio_path)
        if key in self.memory_cache:
            return self.memory_cache[key]

        cache_file = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if cache_file and cache_file.exists():
            feat = np.load(cache_file)
            self.memory_cache[key] = feat
            return feat

        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        if self.max_audio_seconds is not None:
            y = y[: int(self.max_audio_seconds * self.sample_rate)]
        if y.size == 0:
            raise RuntimeError(f"empty audio: {audio_path}")

        inputs = self.processor(
            y,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)

        with self.torch.no_grad():
            outputs = self.model(input_values)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hs = outputs.hidden_states
            layer = self.layer if self.layer >= 0 else len(hs) + self.layer
            if layer < 0 or layer >= len(hs):
                raise ValueError(f"invalid --layer {self.layer}; hidden_states length={len(hs)}")
            feat = hs[layer][0].detach().cpu().float().numpy()
        elif hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state[0].detach().cpu().float().numpy()
        else:
            raise RuntimeError("MERT output has neither hidden_states nor last_hidden_state")

        feat = cosine_normalize(feat.astype(np.float32))

        if cache_file:
            np.save(cache_file, feat)
        self.memory_cache[key] = feat
        return feat


def compute_pair(src_audio: str, edit_audio: str, extractor: MERTFeatureExtractor, quantile: float) -> Dict[str, float]:
    src_feat = extractor.extract(src_audio)
    edit_feat = extractor.extract(edit_audio)

    path, path_sims, total_cost = exact_dtw_from_cosine(src_feat, edit_feat)

    detac = float(np.mean(path_sims))
    q_value = float(np.quantile(path_sims, quantile))

    return {
        "deTAC": detac,
        f"deTAC_q{quantile:g}": q_value,
        "deTAC_min": float(np.min(path_sims)),
        "deTAC_median": float(np.median(path_sims)),
        "deTAC_std": float(np.std(path_sims)),
        "dtw_cost_total": float(total_cost),
        "dtw_cost_mean": float(1.0 - detac),
        "dtw_path_len": int(len(path_sims)),
        "src_frames": int(src_feat.shape[0]),
        "edit_frames": int(edit_feat.shape[0]),
        "feature_dim": int(src_feat.shape[1]),
    }


def write_summary(df: pd.DataFrame, out_summary: Path):
    ok_df = df[df["eval_status"] == "ok"].copy()
    if len(ok_df) == 0:
        print("[WARN] no ok rows, summary skipped")
        return

    metric_cols = [
        c for c in ok_df.columns
        if c.startswith("deTAC") or c.startswith("dtw_") or c in {"src_frames", "edit_frames", "feature_dim"}
    ]
    summary = ok_df[metric_cols].agg(["count", "mean", "std", "min", "median", "max"]).T
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary)
    print("[INFO] summary saved:", out_summary)
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="generation_manifest.csv")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_summary", default=None)
    parser.add_argument("--model_name", default="m-a-p/MERT-v1-95M", help="MERT HF name or local path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=-1, help="MERT hidden layer; -1 means last layer")
    parser.add_argument("--quantile", type=float, default=0.1, help="low quantile for local worst sections")
    parser.add_argument("--cache_dir", default="./eval/.detac_mert_cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_audio_seconds", type=float, default=None, help="debug only: crop audio before feature extraction")
    parser.add_argument("--source_audio_col", default="source_audio")
    parser.add_argument("--edited_audio_col", default="edited_audio")
    parser.add_argument("--status_col", default="status")
    parser.add_argument("--ok_status", default="ok")
    args = parser.parse_args()

    if not (0.0 < args.quantile < 1.0):
        raise ValueError("--quantile must be in (0, 1), e.g. 0.1")

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    out_csv = Path(args.out_csv) if args.out_csv else manifest.parent / "eval_deTAC.csv"
    out_summary = Path(args.out_summary) if args.out_summary else manifest.parent / "eval_deTAC_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    for col in [args.source_audio_col, args.edited_audio_col]:
        if col not in df.columns:
            raise RuntimeError(f"manifest missing column {col}; columns={df.columns.tolist()}")

    if args.status_col in df.columns:
        df = df[df[args.status_col] == args.ok_status].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("[INFO] manifest:", manifest)
    print("[INFO] rows to evaluate:", len(df))
    print("[INFO] out_csv:", out_csv)

    extractor = MERTFeatureExtractor(
        model_name=args.model_name,
        device=args.device,
        layer=args.layer,
        cache_dir=args.cache_dir,
        max_audio_seconds=args.max_audio_seconds,
    )

    rows = []
    fail = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="deTAC eval"):
        item = row.to_dict()
        src_audio = str(row[args.source_audio_col])
        edit_audio = str(row[args.edited_audio_col])

        if not Path(src_audio).exists():
            item.update({"eval_status": "missing_source_audio", "eval_error": f"not found: {src_audio}"})
            rows.append(item)
            fail += 1
            continue
        if not Path(edit_audio).exists():
            item.update({"eval_status": "missing_edited_audio", "eval_error": f"not found: {edit_audio}"})
            rows.append(item)
            fail += 1
            continue

        try:
            metrics = compute_pair(src_audio, edit_audio, extractor, args.quantile)
            item.update(metrics)
            item.update({"eval_status": "ok", "eval_error": ""})
        except Exception as e:
            item.update({"eval_status": "eval_failed", "eval_error": repr(e)})
            fail += 1

        rows.append(item)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    print("[INFO] eval finished")
    print("[INFO] ok:", int((out["eval_status"] == "ok").sum()) if "eval_status" in out.columns else 0)
    print("[INFO] fail:", fail)
    print("[INFO] saved:", out_csv)

    write_summary(out, out_summary)


if __name__ == "__main__":
    main()
