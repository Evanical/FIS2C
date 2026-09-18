#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py

Objective metric evaluator adapted for batch_edit_fisc outputs.

Supported modes:
1) --manifest existing_manifest.csv
   CSV must contain source_audio and edited_audio.

2) --batch_output_root ./outputs/xxx
   The script will read:
     batch_status.jsonl
     batch_status_worker*.jsonl
   and find edited audio under output_dir.

3) Robust fallback scan:
   If status jsonl is missing, incomplete, or only contains skipped_existing rows,
   the script scans subfolders like:
     00000_xxx/xxx_fisc_best.wav
   and reconstructs source audio/prompt from --metadata + --audio_root when provided.

Example:
python eval/test.py \
  --batch_output_root ./outputs/batch_fisc_step300_strength13_full \
  --metadata /mnt/westdata/lmy/data/ZoME-Bench/metadata_with_audio_abs.parquet \
  --audio_root /mnt/westdata/lmy/data/ZoME-Bench/audio \
  --out_csv ./outputs/batch_fisc_step300_strength13_full/eval_metrics.csv \
  --metrics clap lpaps cqt
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
EVAL_DIR = THIS_FILE.parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------



def patch_instance_get_audio_features_kwargs(obj):
    """
    Instance-level compatibility patch for laion_clap / LPAPS.

    Fixes:
      TypeError: get_audio_features() got an unexpected keyword argument 'require_grad'
      AttributeError: 'CLAP_Module' object has no attribute 'enable_fusion'
    """
    import inspect

    visited = set()
    patched = 0

    def safe_setattr(x, name, value):
        try:
            if not hasattr(x, name):
                setattr(x, name, value)
        except Exception:
            pass

    def wrap(owner, name):
        nonlocal patched
        try:
            old = getattr(owner, name)
        except Exception:
            return
        if old is None or getattr(old, "_lpaps_drop_kwargs_patch", False):
            return

        def wrapped(*args, **kwargs):
            kwargs.pop("require_grad", None)
            kwargs.pop("enable_fusion", None)
            kwargs.pop("use_tensor", None)
            try:
                return old(*args, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    try:
                        sig = inspect.signature(old)
                        allowed = set(sig.parameters.keys())
                        kwargs2 = {k: v for k, v in kwargs.items() if k in allowed}
                        return old(*args, **kwargs2)
                    except Exception:
                        return old(*args)
                raise

        wrapped._lpaps_drop_kwargs_patch = True
        try:
            setattr(owner, name, wrapped)
            patched += 1
        except Exception:
            pass

    def visit(x, depth=0):
        if x is None or depth > 10 or id(x) in visited:
            return
        visited.add(id(x))

        safe_setattr(x, "enable_fusion", False)
        safe_setattr(x, "require_grad", False)

        for method_name in [
            "get_audio_features",
            "get_audio_embedding_from_data",
            "get_audio_embedding",
            "get_text_embedding",
        ]:
            wrap(x, method_name)

        try:
            values = list(vars(x).values())
        except Exception:
            values = []

        # Named attributes first.
        for attr in [
            "model", "net", "module", "clap", "clap_model", "audio_model",
            "audio_branch", "text_branch", "encoder", "base", "backbone"
        ]:
            try:
                values.append(getattr(x, attr))
            except Exception:
                pass

        for v in values:
            if isinstance(v, (str, bytes, int, float, bool, type(None))):
                continue
            if isinstance(v, dict):
                for vv in v.values():
                    visit(vv, depth + 1)
            elif isinstance(v, (list, tuple, set)):
                for vv in v:
                    visit(vv, depth + 1)
            else:
                visit(v, depth + 1)

        try:
            if hasattr(x, "modules"):
                for m in x.modules():
                    visit(m, depth + 1)
        except Exception:
            pass

    visit(obj)
    print(f"[INFO] instance patched get_audio_features kwargs on {patched} methods")


def lazy_torch():
    import torch
    import torchaudio
    return torch, torchaudio


def patch_clap_use_tensor_api(obj, device="cpu", _depth=0, _seen=None):
    """
    Robust compatibility patch for mixed LPAPS / laion_clap versions.

    It fixes the common failures seen in this environment:
      1) missing use_tensor kwarg support;
      2) CUDA tensor -> numpy conversion inside old laion_clap embedding APIs;
      3) get_audio_features(..., require_grad=...) TypeError;
      4) get_audio_features(..., use_tensor=...) TypeError.

    Important:
    - get_audio_embedding_from_data / get_text_embedding are patched with CPU numpy conversion.
    - get_audio_features is patched ONLY to drop unsupported kwargs, keeping tensor inputs unchanged.
    """
    if obj is None:
        return
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    def _to_cpu_numpy(x):
        try:
            import torch
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(x, (list, tuple)):
            return type(x)(_to_cpu_numpy(v) for v in x)
        if isinstance(x, dict):
            return {k: _to_cpu_numpy(v) for k, v in x.items()}
        return x

    def _patch_embedding_method(target, method_name):
        if not hasattr(target, method_name):
            return
        try:
            orig = getattr(target, method_name)
        except Exception:
            return
        if getattr(orig, "_fisc_compat_patch", False):
            return

        def wrapped(*args, **kwargs):
            use_tensor = kwargs.pop("use_tensor", None)
            kwargs.pop("require_grad", None)
            args2 = tuple(_to_cpu_numpy(a) for a in args)
            kwargs2 = {k: _to_cpu_numpy(v) for k, v in kwargs.items()}
            out = orig(*args2, **kwargs2)
            if use_tensor is True:
                import torch
                if torch.is_tensor(out):
                    return out.to(device)
                return torch.as_tensor(out, device=device)
            return out

        wrapped._fisc_compat_patch = True
        try:
            setattr(target, method_name, wrapped)
            print(f"[INFO] patched {target.__class__.__name__}.{method_name} embedding compatibility")
        except Exception:
            pass

    def _patch_audio_features_method(target):
        method_name = "get_audio_features"
        if not hasattr(target, method_name):
            return
        try:
            orig = getattr(target, method_name)
        except Exception:
            return
        if getattr(orig, "_fisc_compat_patch", False):
            return

        def wrapped(*args, **kwargs):
            # Different laion_clap versions disagree on these kwargs.
            kwargs.pop("require_grad", None)
            kwargs.pop("use_tensor", None)
            return orig(*args, **kwargs)

        wrapped._fisc_compat_patch = True
        try:
            setattr(target, method_name, wrapped)
            print(f"[INFO] patched {target.__class__.__name__}.get_audio_features drop unsupported kwargs")
        except Exception:
            pass

    # Patch current object.
    _patch_embedding_method(obj, "get_audio_embedding_from_data")
    _patch_embedding_method(obj, "get_text_embedding")
    _patch_audio_features_method(obj)

    # Traverse all torch modules if available. This catches CLAP_Module buried inside LPAPS.net.
    try:
        for m in obj.modules():
            if m is not obj:
                patch_clap_use_tensor_api(m, device=device, _seen=_seen)
    except Exception:
        pass

    # Traverse common wrapper attributes.
    for name in [
        "model", "net", "module", "clap", "clap_model", "audio_model",
        "audio_branch", "caption_branch", "base", "encoder", "module_list"
    ]:
        try:
            child = getattr(obj, name, None)
        except Exception:
            child = None
        if child is not None and child is not obj:
            patch_clap_use_tensor_api(child, device=device, _seen=_seen)

    # Traverse arbitrary attributes and containers. This catches CLAP_Module
    # stored under names not listed above.
    try:
        vals = list(vars(obj).values())
    except Exception:
        vals = []
    for val in vals:
        if val is obj:
            continue
        if isinstance(val, dict):
            children = list(val.values())
        elif isinstance(val, (list, tuple, set)):
            children = list(val)
        else:
            children = [val]
        for child in children:
            try:
                if hasattr(child, "__dict__") or hasattr(child, "modules"):
                    patch_clap_use_tensor_api(child, device=device, _seen=_seen)
            except Exception:
                pass


def patch_missing_enable_fusion_attr(obj, _depth=0, _seen=None):
    """
    Some laion_clap versions have CLAP_Module.forward() reading
    self.enable_fusion, but the attribute is never created for non-fusion
    checkpoints. Add enable_fusion=False recursively to prevent every
    LPAPS window from failing with:
      AttributeError: 'CLAP_Module' object has no attribute 'enable_fusion'
    """
    if obj is None or _depth > 10:
        return
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    try:
        if not hasattr(obj, "enable_fusion"):
            setattr(obj, "enable_fusion", False)
    except Exception:
        pass

    # Traverse torch modules.
    try:
        for child in obj.children():
            patch_missing_enable_fusion_attr(child, _depth + 1, _seen)
    except Exception:
        pass

    # Traverse common wrapper attributes used in LPAPS / CLAP.
    for name in ["model", "net", "module", "clap", "clap_model", "audio_model", "audio_branch", "caption_branch"]:
        try:
            child = getattr(obj, name, None)
        except Exception:
            child = None
        if child is not None and child is not obj:
            patch_missing_enable_fusion_attr(child, _depth + 1, _seen)


def patch_global_get_audio_features_kwargs():
    """
    Global monkey-patch for laion_clap / CLAP_Module API mismatch.
    Some LPAPS/pretrained_networks versions call:
        get_audio_features(..., require_grad=False)
    while the installed CLAP version does not accept require_grad.
    This patches every loaded class that defines get_audio_features so the
    unsupported kwargs are removed before the original method is called.
    """
    import sys
    import inspect

    patched = 0
    seen = set()

    def patch_class(cls):
        nonlocal patched
        try:
            orig = getattr(cls, "get_audio_features", None)
        except Exception:
            return
        if orig is None or getattr(orig, "_drop_bad_lpaps_kwargs", False):
            return
        key = (id(cls), id(orig))
        if key in seen:
            return
        seen.add(key)

        def wrapped(self, *args, **kwargs):
            kwargs.pop("require_grad", None)
            kwargs.pop("use_tensor", None)
            return orig(self, *args, **kwargs)

        wrapped._drop_bad_lpaps_kwargs = True
        try:
            setattr(cls, "get_audio_features", wrapped)
            patched += 1
        except Exception:
            pass

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            values = list(vars(mod).values())
        except Exception:
            continue
        for v in values:
            try:
                if inspect.isclass(v) and hasattr(v, "get_audio_features"):
                    patch_class(v)
            except Exception:
                pass

    print(f"[INFO] globally patched get_audio_features kwargs on {patched} classes")

def safe_exists(path: str) -> bool:
    try:
        return Path(str(path)).exists()
    except Exception:
        return False


def to_float(x):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return np.nan


def pearson_value(result):
    if hasattr(result, "statistic"):
        return float(result.statistic)
    if isinstance(result, (tuple, list)) and len(result) > 0:
        return float(result[0])
    return float(result)


def is_empty_value(x) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def first_existing_file(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return str(Path(matches[0]).resolve())
    return None


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".json":
        # Support both normal JSON array and JSONL/NDJSON saved with .json suffix.
        # MusicBench_test_A.json is JSONL: one JSON object per line.
        try:
            return pd.read_json(p)
        except ValueError as e:
            if "Trailing data" not in str(e):
                raise
            rows = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return pd.DataFrame(rows)
    if suffix in [".jsonl", ".ndjson"]:
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported metadata format: {p}")


def get_metadata_row_by_original_index(metadata_df: pd.DataFrame, idx: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    Resolve a metadata row from either:
      1) full metadata, where idx is the row position; or
      2) split metadata, where idx should match original_index.

    This prevents paper460/train/val split files from breaking fallback folder
    scanning when output folders are named by original metadata index.
    """
    if metadata_df is None or idx is None:
        return None

    if "original_index" in metadata_df.columns:
        original_index = pd.to_numeric(metadata_df["original_index"], errors="coerce")
        matches = metadata_df[original_index == int(idx)]
        if len(matches) > 0:
            return matches.iloc[0].to_dict()

    if 0 <= int(idx) < len(metadata_df):
        return metadata_df.iloc[int(idx)].to_dict()

    return None


# ---------------------------------------------------------------------
# Build manifest from batch outputs
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] failed to parse jsonl line in {path}: {repr(e)}")
    return rows


def find_status_files(batch_output_root: Path, explicit_status: Optional[str] = None) -> List[Path]:
    if explicit_status:
        files = [Path(x).resolve() for x in explicit_status.split(",") if x.strip()]
        return [p for p in files if p.exists()]

    candidates = []
    for name in [
        "batch_status.jsonl",
        "batch_status_worker*.jsonl",
        "**/batch_status.jsonl",
        "**/batch_status_worker*.jsonl",
    ]:
        candidates.extend(batch_output_root.glob(name))

    files = sorted({str(p.resolve()): p.resolve() for p in candidates}.values())
    return files


def resolve_edited_audio_from_record(
    record: Dict[str, Any],
    batch_output_root: Path,
    edited_variant: str = "best",
    edited_glob: Optional[str] = None,
) -> Optional[str]:
    if edited_variant == "best":
        for key in ["best_path", "edited_audio", "fisc_best", "output_audio"]:
            if key in record and record[key] and safe_exists(record[key]):
                return str(Path(record[key]).resolve())
    elif edited_variant == "last":
        for key in ["last_path", "edited_audio", "fisc_last", "output_audio"]:
            if key in record and record[key] and safe_exists(record[key]):
                return str(Path(record[key]).resolve())

    output_dir = record.get("output_dir", "")
    if not output_dir:
        return None

    out_dir = Path(str(output_dir))
    if not out_dir.is_absolute():
        out_dir = batch_output_root / out_dir

    if edited_glob:
        patterns = [str(out_dir / edited_glob)]
    else:
        if edited_variant == "last":
            patterns = [
                str(out_dir / "*_fisc_last.wav"),
                str(out_dir / "*_baseline_last.wav"),
                str(out_dir / "*last*.wav"),
            ]
        else:
            patterns = [
                str(out_dir / "*_fisc_best.wav"),
                str(out_dir / "*_baseline_best.wav"),
                str(out_dir / "*best*.wav"),
                str(out_dir / "*.wav"),
            ]

    return first_existing_file(patterns)


def parse_index_from_folder(folder: Path) -> Optional[int]:
    m = re.match(r"^(\d+)_", folder.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def folder_stem_without_index(folder: Path) -> str:
    return re.sub(r"^\d+_", "", folder.name)


def resolve_audio_from_metadata_row(
    row: Dict[str, Any],
    metadata_dir: Path,
    audio_root: Optional[Path],
    audio_key: str,
    ytid_key: str,
    folder_hint: str = "",
) -> Optional[str]:
    candidates = []

    if audio_key in row and not is_empty_value(row[audio_key]):
        raw = str(row[audio_key]).strip()
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            if audio_root is not None:
                candidates.append(audio_root / p)
                candidates.append(audio_root / p.name)
            candidates.append(metadata_dir / p)
            candidates.append(metadata_dir / "audio" / p)
            candidates.append(metadata_dir / "audio" / p.name)

    if audio_root is not None and ytid_key in row and not is_empty_value(row[ytid_key]):
        ytid = str(row[ytid_key]).strip()
        candidates.extend(sorted(audio_root.glob(f"{ytid}_*.wav")))

    # Last-resort: use folder/file stem. This handles sanitized folders where leading '-' was stripped.
    if audio_root is not None and folder_hint:
        h = folder_hint.strip()
        candidates.append(audio_root / f"{h}.wav")
        candidates.append(audio_root / f"-{h}.wav")
        candidates.extend(sorted(audio_root.glob(f"*{h}.wav")))
        candidates.extend(sorted(audio_root.glob(f"*{h}*.wav")))

    seen = set()
    for c in candidates:
        c = Path(c)
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return str(c.resolve())

    return None


def scan_manifest_from_folders(
    batch_output_root: Path,
    edited_variant: str,
    edited_glob: Optional[str],
    metadata: Optional[str],
    audio_root: Optional[str],
    source_prompt_key: str,
    target_prompt_key: str,
    audio_key: str,
    ytid_key: str,
    limit: Optional[int],
) -> pd.DataFrame:
    print("[INFO] fallback: scanning output folders directly...")

    metadata_df = None
    metadata_dir = None
    if metadata:
        metadata_path = Path(metadata).resolve()
        metadata_df = read_table(str(metadata_path))
        metadata_dir = metadata_path.parent
        print("[INFO] loaded metadata for fallback:", metadata_path, metadata_df.shape)
    else:
        print("[WARN] --metadata not provided; CLAP prompts may be empty, and source_audio is resolved only from --audio_root + folder name.")

    audio_root_path = Path(audio_root).resolve() if audio_root else None

    rows = []
    folders = sorted([p for p in batch_output_root.iterdir() if p.is_dir()])

    for folder in folders:
        idx = parse_index_from_folder(folder)
        hint = folder_stem_without_index(folder)

        if edited_glob:
            edited = first_existing_file([str(folder / edited_glob)])
        else:
            if edited_variant == "last":
                edited = first_existing_file([str(folder / "*_fisc_last.wav"), str(folder / "*last*.wav")])
            else:
                edited = first_existing_file([str(folder / "*_fisc_best.wav"), str(folder / "*best*.wav"), str(folder / "*.wav")])

        if not edited:
            continue

        source_prompt = ""
        target_prompt = ""
        source_audio = None

        if metadata_df is not None and idx is not None:
            mrow = get_metadata_row_by_original_index(metadata_df, idx)
            if mrow is not None:
                source_prompt = str(mrow.get(source_prompt_key, ""))
                target_prompt = str(mrow.get(target_prompt_key, ""))
                source_audio = resolve_audio_from_metadata_row(
                    mrow,
                    metadata_dir=metadata_dir,
                    audio_root=audio_root_path,
                    audio_key=audio_key,
                    ytid_key=ytid_key,
                    folder_hint=hint,
                )

        if source_audio is None and audio_root_path is not None:
            # folder hint, plus the edited wav stem minus suffix
            hints = [hint]
            edited_stem = Path(edited).stem
            edited_stem = re.sub(r"_(fisc|baseline)_(best|last)$", "", edited_stem)
            hints.append(edited_stem.lstrip("-"))
            for h in hints:
                source_audio = resolve_audio_from_metadata_row(
                    {},
                    metadata_dir=batch_output_root,
                    audio_root=audio_root_path,
                    audio_key=audio_key,
                    ytid_key=ytid_key,
                    folder_hint=h,
                )
                if source_audio:
                    break

        if not source_audio or not safe_exists(source_audio):
            continue

        rows.append({
            "index": idx if idx is not None else len(rows),
            "sample_name": folder.name,
            "status": "ok",
            "source_audio": str(Path(source_audio).resolve()),
            "edited_audio": str(Path(edited).resolve()),
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "output_dir": str(folder.resolve()),
            "batch_status_file": "",
            "manifest_source": "folder_scan",
        })

    df = pd.DataFrame(rows)
    if limit is not None:
        df = df.head(limit).copy()

    return df


def build_manifest_from_batch_outputs(
    batch_output_root: str,
    batch_status: Optional[str] = None,
    edited_variant: str = "best",
    edited_glob: Optional[str] = None,
    only_ok: bool = True,
    limit: Optional[int] = None,
    metadata: Optional[str] = None,
    audio_root: Optional[str] = None,
    source_prompt_key: str = "original_prompt",
    target_prompt_key: str = "editing_prompt",
    audio_key: str = "audio_path",
    ytid_key: str = "ytid",
) -> pd.DataFrame:
    root = Path(batch_output_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"batch_output_root does not exit: {root}")

    status_files = find_status_files(root, batch_status)
    print("[INFO] batch_output_root:", root)
    print("[INFO] status files:")
    for p in status_files:
        print("  ", p)

    rows = []
    seen = set()

    valid_status = {"ok", "skipped_existing", "dry_run", ""}

    for status_path in status_files:
        for rec in read_jsonl(status_path):
            status = str(rec.get("status", ""))
            if only_ok and status not in valid_status:
                continue

            source_audio = rec.get("audio_path") or rec.get("source_audio")
            source_prompt = rec.get("source_prompt", "")
            target_prompt = rec.get("target_prompt", "")
            output_dir = rec.get("output_dir", "")

            if not source_audio or not safe_exists(source_audio):
                continue

            edited_audio = resolve_edited_audio_from_record(
                rec,
                batch_output_root=root,
                edited_variant=edited_variant,
                edited_glob=edited_glob,
            )
            if not edited_audio or not safe_exists(edited_audio):
                continue

            key = (str(Path(source_audio).resolve()), str(Path(edited_audio).resolve()))
            if key in seen:
                continue
            seen.add(key)

            item = {
                "index": rec.get("index", len(rows)),
                "sample_name": rec.get("sample_name", Path(edited_audio).parent.name),
                "status": "ok",
                "source_audio": str(Path(source_audio).resolve()),
                "edited_audio": str(Path(edited_audio).resolve()),
                "source_prompt": str(source_prompt),
                "target_prompt": str(target_prompt),
                "output_dir": str(Path(output_dir).resolve()) if output_dir else str(Path(edited_audio).parent.resolve()),
                "batch_status_file": str(status_path),
                "manifest_source": "batch_status",
            }

            for k in ["best_score", "text_cache_size", "returncode", "error"]:
                if k in rec:
                    item[k] = rec[k]

            rows.append(item)

    df = pd.DataFrame(rows)

    if df.empty:
        df = scan_manifest_from_folders(
            batch_output_root=root,
            edited_variant=edited_variant,
            edited_glob=edited_glob,
            metadata=metadata,
            audio_root=audio_root,
            source_prompt_key=source_prompt_key,
            target_prompt_key=target_prompt_key,
            audio_key=audio_key,
            ytid_key=ytid_key,
            limit=limit,
        )
    elif limit is not None:
        df = df.head(limit).copy()

    return df


# ---------------------------------------------------------------------
# Metric scorers
# ---------------------------------------------------------------------

class CLAPScorer:
    def __init__(self, device: str, checkpoint_dir: str, checkpoint_name: str):
        argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            import torch
            from meta_clap_consistency import CLAPTextConsistencyMetric

            self.torch = torch
            self.device = device
            self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(f"CLAP checkpoint does not exist: {self.checkpoint_path}")

            self.model = CLAPTextConsistencyMetric(
                model_path=self.checkpoint_path,
                model_arch="HTSAT-base" if "fusion" not in checkpoint_name else "HTSAT-tiny",
                enable_fusion="fusion" in checkpoint_name,
            ).to(device)
            patch_global_get_audio_features_kwargs()
            patch_instance_get_audio_features_kwargs(self.model)
            self.model.eval()
            patch_missing_enable_fusion_attr(self.model)
            patch_clap_use_tensor_api(self.model, device=device)
            print("[INFO] LPAPS safe compatibility patches applied")
        finally:
            sys.argv = argv_backup

    def score_audio_text(self, audio_path: str, prompt: str) -> float:
        torch, torchaudio = lazy_torch()

        if prompt is None or str(prompt).strip() == "":
            return np.nan

        aud, sr = torchaudio.load(audio_path)

        with torch.no_grad():
            self.model.update(
                aud.unsqueeze(0).to(self.device),
                [str(prompt)],
                torch.tensor([sr], device=self.device),
            )
            score = self.model.compute()
            self.model.reset()

        return to_float(score)


class LPAPSScorer:
    """
    Safer LPAPS scorer.

    Changes compared with the original version:
    1) Converts stereo/multi-channel audio to mono before LPAPS.
    2) Removes non-finite window scores instead of letting one bad window poison the whole result.
    3) Records the last LPAPS error / number of valid windows for debugging.
    4) Raises a clear error when every LPAPS window failed, so *_error in CSV is useful.
    """

    def __init__(
        self,
        device: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        method: str = "mean",
        overlap: float = 0.1,
        win_length: Optional[int] = 10,
    ):
        argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            import torch
            from lpaps import LPAPS
            patch_global_get_audio_features_kwargs()

            self.torch = torch
            self.device = device
            self.method = method
            self.overlap = overlap
            self.win_length = win_length if "fusion" not in checkpoint_name else None
            self.last_error = ""
            self.last_num_windows = 0
            self.last_num_failed_windows = 0

            checkpoint_path = Path(checkpoint_dir) / checkpoint_name
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"LPAPS/CLAP checkpoint 不存在: {checkpoint_path}")

            # IMPORTANT:
            # Some laion_clap versions create a CLAP_Module without the
            # `enable_fusion` attribute. Passing enable_fusion through LPAPS
            # can then make every LPAPS window fail with:
            # AttributeError: 'CLAP_Module' object has no attribute 'enable_fusion'
            #
            # For the normal checkpoint music_audioset_epoch_15_esc_90.14.pt,
            # do NOT pass enable_fusion.
            self.model = LPAPS(
                net="clap",
                device=device,
                net_kwargs={
                    "model_arch": "HTSAT-base",
                    "chkpt": checkpoint_name,
                },
                checkpoint_path=checkpoint_dir,
            ).to(device)
            patch_global_get_audio_features_kwargs()
            self.model.eval()
            patch_missing_enable_fusion_attr(self.model)
            patch_clap_use_tensor_api(self.model, device=device)
            print("[INFO] LPAPS safe compatibility patches applied")
        finally:
            sys.argv = argv_backup

    @staticmethod
    def _to_mono(aud):
        # torchaudio.load returns [channels, time]. LPAPS/CLAP is more stable with mono audio.
        if aud.dim() == 2 and aud.shape[0] > 1:
            aud = aud.mean(dim=0, keepdim=True)
        return aud.float().contiguous()

    @staticmethod
    def _finite_float(x) -> float:
        v = to_float(x)
        if not np.isfinite(v):
            return np.nan
        return float(v)

    def _call_model(self, w1, sr1: int, w2, sr2: int) -> float:
        torch = self.torch
        with torch.no_grad():
            patch_instance_get_audio_features_kwargs(self.model)
            out = self.model(
                w1.unsqueeze(0).to(self.device),
                w2.unsqueeze(0).to(self.device),
                torch.tensor([int(sr1)], device=self.device),
                torch.tensor([int(sr2)], device=self.device),
            )
        return self._finite_float(out)

    def _compute_with_windows(self, aud1, sr1: int, aud2, sr2: int) -> float:
        self.last_error = ""
        self.last_num_windows = 0
        self.last_num_failed_windows = 0

        aud1 = self._to_mono(aud1)
        aud2 = self._to_mono(aud2)

        if aud1.shape[-1] <= 0 or aud2.shape[-1] <= 0:
            raise RuntimeError(f"empty audio tensor: aud1={tuple(aud1.shape)}, aud2={tuple(aud2.shape)}")

        # Fusion checkpoint path: compute the whole clip directly.
        if self.win_length is None:
            score = self._call_model(aud1, sr1, aud2, sr2)
            self.last_num_windows = 1 if np.isfinite(score) else 0
            if not np.isfinite(score):
                raise RuntimeError("LPAPS returned NaN/Inf for full audio")
            return score

        win1 = max(1, int(float(self.win_length) * int(sr1)))
        win2 = max(1, int(float(self.win_length) * int(sr2)))
        step1 = max(1, int(win1 * (1.0 - float(self.overlap))))
        step2 = max(1, int(win2 * (1.0 - float(self.overlap))))

        scores = []
        errors = []

        for i, j in zip(range(0, aud1.shape[-1], step1), range(0, aud2.shape[-1], step2)):
            w1 = aud1[:, i:i + win1]
            w2 = aud2[:, j:j + win2]
            if w1.shape[-1] == 0 or w2.shape[-1] == 0:
                continue

            # Very tiny tail windows often make CLAP/LPAPS unstable. Skip tails shorter than 0.25 s.
            min_len1 = max(1, int(0.25 * int(sr1)))
            min_len2 = max(1, int(0.25 * int(sr2)))
            if w1.shape[-1] < min_len1 or w2.shape[-1] < min_len2:
                continue

            try:
                v = self._call_model(w1, sr1, w2, sr2)
                if np.isfinite(v):
                    scores.append(v)
                else:
                    errors.append("LPAPS window returned NaN/Inf")
            except Exception as e:
                errors.append(repr(e))

        self.last_num_windows = len(scores)
        self.last_num_failed_windows = len(errors)
        if errors:
            self.last_error = errors[0]

        if not scores:
            detail = self.last_error or "no valid LPAPS windows"
            raise RuntimeError(
                f"LPAPS failed: no finite scores. "
                f"audio_len=({aud1.shape[-1]}/{sr1:.0f}samp, {aud2.shape[-1]}/{sr2:.0f}samp), "
                f"win=({win1},{win2}), step=({step1},{step2}), first_error={detail}"
            )

        arr = np.asarray(scores, dtype=np.float64)
        if self.method == "mean":
            return float(np.mean(arr))
        if self.method == "median":
            return float(np.median(arr))
        if self.method == "max":
            return float(np.max(arr))
        if self.method == "min":
            return float(np.min(arr))
        raise ValueError(f"Unknown LPAPS method: {self.method}")

    def score_pair(self, source_audio: str, edited_audio: str) -> float:
        _, torchaudio = lazy_torch()
        aud1, sr1 = torchaudio.load(source_audio)
        aud2, sr2 = torchaudio.load(edited_audio)
        return self._compute_with_windows(aud1, int(sr1), aud2, int(sr2))

class CDPAMScorer:
    def __init__(self, device: str):
        import torch
        import cdpam

        self.torch = torch
        self.cdpam = cdpam
        self.loss_fn = cdpam.CDPAM(dev=device if device.startswith("cuda") else "cpu")

    def score_pair(self, source_audio: str, edited_audio: str) -> float:
        y1 = self.cdpam.load_audio(source_audio)
        y2 = self.cdpam.load_audio(edited_audio)
        with self.torch.no_grad():
            return to_float(self.loss_fn.forward(y1, y2))


class CQTScorer:
    def __init__(self, device: str = "cpu", sr: int = 16000):
        import torch
        import torchaudio
        import torchaudio.transforms as T
        from nnAudio.features import CQT2010
        from scipy.stats import pearsonr

        self.torch = torch
        self.torchaudio = torchaudio
        self.T = T
        self.pearsonr = pearsonr
        self.sr = sr
        self.device = device
        self.cqt = CQT2010(
            sr=sr,
            fmin=65,
            fmax=2100,
            n_bins=128,
            bins_per_octave=24,
            norm=1,
            basis_norm=1,
            window="hann",
            pad_mode="constant",
            earlydownsample=True,
        ).to(device)

    def cqt_top1(self, audio_path: str):
        audio, sr_org = self.torchaudio.load(audio_path)
        if sr_org != self.sr:
            audio = self.T.Resample(orig_freq=sr_org, new_freq=self.sr)(audio)
        audio = audio.to(self.device)
        with self.torch.no_grad():
            _, index = self.cqt(audio)[0].topk(4, dim=0)
        return index[0].detach().cpu().numpy()

    def score_pair(self, source_audio: str, edited_audio: str) -> float:
        idx1 = self.cqt_top1(source_audio)
        idx2 = self.cqt_top1(edited_audio)
        n = min(len(idx1), len(idx2))
        if n < 2:
            return np.nan
        return pearson_value(self.pearsonr(idx1[:n], idx2[:n]))


class DeTACScorer:
    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "m-a-p/MERT-v1-95M",
        layer: int = -1,
        quantile: float = 0.1,
        cache_dir: str = "./eval/.detac_mert_cache",
        max_audio_seconds: Optional[float] = None,
    ):
        from deTAC import MERTFeatureExtractor, compute_pair

        self.quantile = quantile
        self.compute_pair = compute_pair
        self.extractor = MERTFeatureExtractor(
            model_name=model_name,
            device=device,
            layer=layer,
            cache_dir=cache_dir,
            max_audio_seconds=max_audio_seconds,
        )

    def score_pair(self, source_audio: str, edited_audio: str) -> Dict[str, float]:
        return self.compute_pair(source_audio, edited_audio, self.extractor, self.quantile)




class ManualCLAPFADScorer:
    """
    Manual CLAP-FAD implementation.

    Why this exists:
    The frechet_audio_distance CLAP backend can return empty embeddings / NaN
    in some environments. This scorer directly uses Hugging Face CLAP audio
    embeddings and computes the Frechet distance locally.

    Requires:
      pip install transformers soundfile scipy
    First run may download:
      laion/clap-htsat-fused
    """

    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "laion/clap-htsat-fused",
        sample_rate: int = 48000,
        batch_size: int = 8,
    ):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.device = device
        self.sample_rate = int(sample_rate)
        self.batch_size = int(batch_size)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def _load_audio_np(self, path: str):
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != self.sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=int(sr), new_freq=self.sample_rate)(wav)
        wav = wav.squeeze(0).float().cpu().numpy()
        return wav

    def embed_paths(self, paths: List[str]) -> np.ndarray:
        embs = []
        torch = self.torch

        with torch.no_grad():
            for i in tqdm(range(0, len(paths), self.batch_size), desc="CLAP audio embeddings", leave=False):
                batch_paths = paths[i:i + self.batch_size]
                audios = [self._load_audio_np(p) for p in batch_paths]

                inputs = self.processor(
                    audios=audios,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if hasattr(self.model, "get_audio_features"):
                    emb = self.model.get_audio_features(**inputs)
                else:
                    out = self.model(**inputs)
                    emb = getattr(out, "audio_embeds", None)
                    if emb is None:
                        raise RuntimeError("CLAP model output does not contain audio embeddings.")

                emb = emb.detach().float().cpu().numpy()
                embs.append(emb)

        if not embs:
            raise RuntimeError("No CLAP embeddings extracted.")

        embs = np.concatenate(embs, axis=0)
        embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
        return embs

    @staticmethod
    def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        from scipy import linalg

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        mu_x = np.mean(x, axis=0)
        mu_y = np.mean(y, axis=0)
        cov_x = np.cov(x, rowvar=False)
        cov_y = np.cov(y, rowvar=False)

        if cov_x.ndim == 0:
            cov_x = np.array([[cov_x]])
        if cov_y.ndim == 0:
            cov_y = np.array([[cov_y]])

        cov_x = cov_x + np.eye(cov_x.shape[0]) * eps
        cov_y = cov_y + np.eye(cov_y.shape[0]) * eps

        diff = mu_x - mu_y
        covmean, _ = linalg.sqrtm(cov_x.dot(cov_y), disp=False)

        if not np.isfinite(covmean).all():
            covmean = linalg.sqrtm((cov_x + np.eye(cov_x.shape[0]) * eps).dot(cov_y + np.eye(cov_y.shape[0]) * eps))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fd = diff.dot(diff) + np.trace(cov_x + cov_y - 2.0 * covmean)
        return float(np.real(fd))

    def score_manifest(self, df: pd.DataFrame) -> float:
        source_paths = [str(p) for p in df["source_audio"].tolist()]
        edited_paths = [str(p) for p in df["edited_audio"].tolist()]
        src_emb = self.embed_paths(source_paths)
        edt_emb = self.embed_paths(edited_paths)
        print("[INFO] manual CLAP source embedding shape:", src_emb.shape)
        print("[INFO] manual CLAP edited embedding shape:", edt_emb.shape)
        return self.frechet_distance(src_emb, edt_emb)


class FADScorer:
    """
    Dataset-level Frechet Audio Distance scorer.

    Requires: pip install frechet-audio-distance

    This version temporarily clears sys.argv because some third-party FAD
    dependencies incorrectly parse the parent script's CLI arguments.
    """
    def __init__(
        self,
        model_name: str = "vggish",
        sample_rate: int = 16000,
        use_pca: bool = False,
        use_activation: bool = False,
        verbose: bool = False,
    ):
        argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            from frechet_audio_distance import FrechetAudioDistance

            kwargs = {
                "model_name": model_name,
                "use_pca": use_pca,
                "use_activation": use_activation,
                "verbose": verbose,
            }
            try:
                self.model = FrechetAudioDistance(sample_rate=sample_rate, **kwargs)
            except TypeError:
                self.model = FrechetAudioDistance(**kwargs)
        finally:
            sys.argv = argv_backup

    def score_dirs(self, source_dir: str, edited_dir: str) -> float:
        argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            return to_float(self.model.score(str(source_dir), str(edited_dir), dtype="float32"))
        finally:
            sys.argv = argv_backup


def _resample_or_copy_for_fad(src: Path, dst: Path, sample_rate: int, force_resample: bool = False):
    """
    Prepare audio for FAD temp directories.

    FAD-CLAP in frechet_audio_distance requires the actual WAV files to be 48 kHz.
    Passing --fad_sample_rate 48000 is not enough if the files themselves are still 16 kHz
    symlinks. Therefore, when sample_rate != 16000 or force_resample=True, we decode
    and save a real resampled wav file instead of symlinking.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    # VGGish default path: keep symlink/copy for speed.
    if (not force_resample) and int(sample_rate) == 16000:
        _safe_link_or_copy(src, dst)
        return

    torch, torchaudio = lazy_torch()
    wav, sr = torchaudio.load(str(src))

    # Convert to mono for FAD consistency.
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if int(sr) != int(sample_rate):
        wav = torchaudio.transforms.Resample(orig_freq=int(sr), new_freq=int(sample_rate))(wav)

    wav = torch.clamp(wav.float(), -1.0, 1.0)
    torchaudio.save(str(dst), wav.cpu(), int(sample_rate))


def _safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(str(src.resolve()), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def prepare_fad_audio_dirs(df: pd.DataFrame, out_csv: Path, args):
    if args.fad_temp_dir:
        root = Path(args.fad_temp_dir)
    else:
        root = out_csv.with_name(out_csv.stem + "_fad_audio_sets")

    root = root.resolve()
    src_dir = root / "source"
    edt_dir = root / "edited"

    if root.exists():
        shutil.rmtree(root)

    src_dir.mkdir(parents=True, exist_ok=True)
    edt_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for _, row in df.iterrows():
        src = Path(str(row["source_audio"]))
        edt = Path(str(row["edited_audio"]))
        if not src.exists() or not edt.exists():
            continue
        _resample_or_copy_for_fad(
            src,
            src_dir / f"{n:06d}.wav",
            sample_rate=int(args.fad_sample_rate),
            force_resample=bool(args.fad_force_resample),
        )
        _resample_or_copy_for_fad(
            edt,
            edt_dir / f"{n:06d}.wav",
            sample_rate=int(args.fad_sample_rate),
            force_resample=bool(args.fad_force_resample),
        )
        n += 1

    if n == 0:
        raise RuntimeError("FAD NaN。")

    return root, src_dir, edt_dir, n


def compute_fad_metrics(out: pd.DataFrame, out_csv: Path, args) -> Dict[str, Any]:
    requested = set(args.metrics)
    want_clap = ("fad" in requested) or ("fad_clap" in requested)
    want_vggish = ("fad" in requested) or ("fad_vggish" in requested)
    if not want_clap and not want_vggish:
        return {}

    results: Dict[str, Any] = {}
    root = None
    try:
        root, src_dir, edt_dir, n = prepare_fad_audio_dirs(out, out_csv, args)
        print("[INFO] FAD audio pairs:", n)
        print("[INFO] FAD source dir:", src_dir)
        print("[INFO] FAD edited dir:", edt_dir)
        print("[INFO] FAD temp audio sample_rate:", args.fad_sample_rate)

        if want_clap:
            try:
                print("[INFO] computing FAD-CLAP...")
                if getattr(args, "fad_clap_backend", "manual") == "manual":
                    print("[INFO] using manual HuggingFace CLAP-FAD backend:", args.fad_clap_hf_model)
                    scorer = ManualCLAPFADScorer(
                        device=args.device,
                        model_name=args.fad_clap_hf_model,
                        sample_rate=48000,
                        batch_size=args.fad_batch_size,
                    )
                    results["fad_clap_source_edited"] = scorer.score_manifest(out)
                else:
                    print("[INFO] using frechet_audio_distance CLAP backend")
                    scorer = FADScorer(
                        model_name=args.fad_clap_model_name,
                        sample_rate=args.fad_sample_rate,
                        use_pca=args.fad_use_pca,
                        use_activation=args.fad_use_activation,
                        verbose=args.fad_verbose,
                    )
                    results["fad_clap_source_edited"] = scorer.score_dirs(src_dir, edt_dir)

                results["fad_clap_status"] = "ok"
                results["fad_clap_error"] = ""
            except Exception as e:
                results["fad_clap_source_edited"] = np.nan
                results["fad_clap_status"] = "failed"
                results["fad_clap_error"] = repr(e)

        if want_vggish:
            try:
                print("[INFO] computing FAD-VGGish...")
                scorer = FADScorer(
                    model_name=args.fad_vggish_model_name,
                    sample_rate=args.fad_sample_rate,
                    use_pca=args.fad_use_pca,
                    use_activation=args.fad_use_activation,
                    verbose=args.fad_verbose,
                )
                results["fad_vggish_source_edited"] = scorer.score_dirs(src_dir, edt_dir)
                results["fad_vggish_status"] = "ok"
                results["fad_vggish_error"] = ""
            except Exception as e:
                results["fad_vggish_source_edited"] = np.nan
                results["fad_vggish_status"] = "failed"
                results["fad_vggish_error"] = repr(e)
    finally:
        if root is not None and (not args.keep_fad_temp):
            try:
                shutil.rmtree(root)
            except Exception:
                pass

    return results



# ---------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------

def load_manifest(path: Path, limit: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    for col in ["source_audio", "edited_audio"]:
        if col not in df.columns:
            raise RuntimeError(f"manifest does not have {col}; 当前列: {df.columns.tolist()}")

    df = df[df["source_audio"].apply(safe_exists) & df["edited_audio"].apply(safe_exists)].copy()

    if limit is not None:
        df = df.head(limit).copy()

    return df


def summarize(df: pd.DataFrame, out_summary: Path):
    """
    Write metric summary and a companion status-count report.

    Important behavior:
    - All numeric metric columns are retained, even if they are all NaN.
      This makes failures such as LPAPS visible in the summary instead of
      silently disappearing.
    - If a metric column is all NaN, its count will be 0 and mean/std/etc.
      will be NaN. Check *_status_counts.csv and *_failed_rows.csv for why.
    """
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {"index", "returncode"}
    ]
    if not numeric_cols:
        print("[WARN] NaN")
        return

    summary = df[numeric_cols].agg(["count", "mean", "std", "min", "median", "max"]).T
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary)
    print("[INFO] summary:", out_summary)
    print(summary)

    all_nan_cols = [c for c in numeric_cols if df[c].isna().all()]
    if all_nan_cols:
        print("[WARN] NaN", all_nan_cols)

    # Save per-metric status counts for quick debugging.
    status_cols = [c for c in df.columns if c.endswith("_status")]
    if status_cols:
        status_rows = []
        for col in status_cols:
            counts = df[col].fillna("<NA>").value_counts(dropna=False)
            metric = col[:-len("_status")]
            for status, count in counts.items():
                status_rows.append({"metric": metric, "status": status, "count": int(count)})
        status_df = pd.DataFrame(status_rows)
        status_path = out_summary.with_name(out_summary.stem + "_status_counts.csv")
        status_df.to_csv(status_path, index=False)
        print("[INFO] status counts:", status_path)

    # Save failed rows with error messages for inspection.
    error_cols = [c for c in df.columns if c.endswith("_error")]
    if error_cols:
        mask = pd.Series(False, index=df.index)
        for col in error_cols:
            mask = mask | df[col].fillna("").astype(str).ne("")
        if mask.any():
            useful_cols = [c for c in ["index", "sample_name", "source_audio", "edited_audio"] if c in df.columns]
            failed_path = out_summary.with_name(out_summary.stem + "_failed_rows.csv")
            df.loc[mask, useful_cols + status_cols + error_cols].to_csv(failed_path, index=False)
            print("[INFO] failed rows:", failed_path)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Old manifest mode
    parser.add_argument("--manifest", default=None, help="CSV with source_audio and edited_audio columns")

    # New batch output mode
    parser.add_argument("--batch_output_root", default=None, help="batch_edit_fisc output root")
    parser.add_argument("--batch_status", default=None, help="optional comma-separated batch_status jsonl paths")
    parser.add_argument("--metadata", default=None, help="metadata parquet/csv/json used to reconstruct source audio/prompt if status jsonl is incomplete")
    parser.add_argument("--audio_root", default=None, help="audio root used to reconstruct source audio if needed")
    parser.add_argument("--source_prompt_key", default="original_prompt")
    parser.add_argument("--target_prompt_key", default="editing_prompt")
    parser.add_argument("--audio_key", default="audio_path")
    parser.add_argument("--ytid_key", default="ytid")

    parser.add_argument("--edited_variant", default="best", choices=["best", "last"])
    parser.add_argument("--edited_glob", default=None, help='custom glob inside each output_dir, e.g. "*_fisc_best.wav"')
    parser.add_argument("--save_manifest", default=None, help="where to save the auto-built manifest from batch outputs")

    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["clap", "lpaps", "cqt", "cdpam", "detac", "fad"],
        choices=["clap", "lpaps", "cqt", "cdpam", "detac", "fad", "fad_clap", "fad_vggish"],
        help=(
            "Metrics to compute. Default intentionally excludes lpaps because LPAPS often fails/returns NaN "
            "with recent laion_clap/torch combinations. Use --metrics ... lpaps only if your LPAPS environment is verified."
        ),
    )
    parser.add_argument("--clap_checkpoint_dir", default="clap/pretrained")
    parser.add_argument("--clap_checkpoint_name", default="music_audioset_epoch_15_esc_90.14.pt")
    parser.add_argument("--lpaps_method", default="mean", choices=["mean", "median", "max", "min"])
    parser.add_argument("--cqt_device", default="cpu")

    # FAD is dataset-level. Requires: pip install frechet-audio-distance
    parser.add_argument("--fad_sample_rate", type=int, default=16000)
    parser.add_argument("--fad_force_resample", action="store_true", help="force writing real resampled wav files for FAD temp folders")
    parser.add_argument("--fad_temp_dir", default=None, help="temporary directory for FAD source/edited audio sets")
    parser.add_argument("--keep_fad_temp", action="store_true", help="keep symlinked FAD temp folders")
    parser.add_argument("--fad_verbose", action="store_true")
    parser.add_argument("--fad_use_pca", action="store_true")
    parser.add_argument("--fad_use_activation", action="store_true")
    parser.add_argument("--fad_clap_model_name", default="clap")
    parser.add_argument("--fad_vggish_model_name", default="vggish")
    parser.add_argument("--fad_clap_backend", default="manual", choices=["manual", "package"], help="manual uses HuggingFace CLAP embeddings; package uses frechet_audio_distance CLAP backend")
    parser.add_argument("--fad_clap_hf_model", default="laion/clap-htsat-fused")
    parser.add_argument("--fad_batch_size", type=int, default=8)

    parser.add_argument("--detac_model_name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--detac_layer", type=int, default=-1)
    parser.add_argument("--detac_quantile", type=float, default=0.1)
    parser.add_argument("--detac_cache_dir", default="./eval/.detac_mert_cache")
    parser.add_argument("--detac_max_audio_seconds", type=float, default=None)

    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary) if args.out_summary else out_csv.with_name(out_csv.stem + "_summary.csv")

    if args.batch_output_root:
        df = build_manifest_from_batch_outputs(
            batch_output_root=args.batch_output_root,
            batch_status=args.batch_status,
            edited_variant=args.edited_variant,
            edited_glob=args.edited_glob,
            only_ok=True,
            limit=args.limit,
            metadata=args.metadata,
            audio_root=args.audio_root,
            source_prompt_key=args.source_prompt_key,
            target_prompt_key=args.target_prompt_key,
            audio_key=args.audio_key,
            ytid_key=args.ytid_key,
        )

        if df.empty:
            raise RuntimeError(
                "--metadata /mnt/westdata/lmy/data/ZoME-Bench/metadata_with_audio_abs.parquet "
                "--audio_root /mnt/westdata/lmy/data/ZoME-Bench/audio。"
            )

        save_manifest = Path(args.save_manifest) if args.save_manifest else out_csv.with_name(out_csv.stem + "_manifest.csv")
        save_manifest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_manifest, index=False)
        print("[INFO] auto-built manifest:", save_manifest)
    elif args.manifest:
        df = load_manifest(Path(args.manifest), args.limit)
    else:
        raise RuntimeError(" --manifest 或 --batch_output_root")

    print("[INFO] rows:", len(df))
    print("[INFO] metrics:", args.metrics)
    print("[INFO] manifest_source counts:")
    if "manifest_source" in df.columns:
        print(df["manifest_source"].value_counts(dropna=False))

    scorers: Dict[str, Any] = {}

    if "clap" in args.metrics:
        print("[INFO] loading CLAP scorer...")
        scorers["clap"] = CLAPScorer(args.device, args.clap_checkpoint_dir, args.clap_checkpoint_name)

    if "lpaps" in args.metrics:
        print("[INFO] loading LPAPS scorer...")
        scorers["lpaps"] = LPAPSScorer(
            args.device,
            args.clap_checkpoint_dir,
            args.clap_checkpoint_name,
            method=args.lpaps_method,
        )

    if "cqt" in args.metrics:
        print("[INFO] loading CQT scorer...")
        scorers["cqt"] = CQTScorer(device=args.cqt_device)

    if "cdpam" in args.metrics:
        print("[INFO] loading CDPAM scorer...")
        scorers["cdpam"] = CDPAMScorer(args.device)

    if "detac" in args.metrics:
        print("[INFO] loading deTAC scorer...")
        scorers["detac"] = DeTACScorer(
            device=args.device,
            model_name=args.detac_model_name,
            layer=args.detac_layer,
            quantile=args.detac_quantile,
            cache_dir=args.detac_cache_dir,
            max_audio_seconds=args.detac_max_audio_seconds,
        )

    rows: List[Dict[str, Any]] = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="external eval"):
        item = row.to_dict()
        src_audio = str(row["source_audio"])
        edt_audio = str(row["edited_audio"])
        src_prompt = str(row.get("source_prompt", ""))
        tgt_prompt = str(row.get("target_prompt", ""))

        if "clap" in scorers:
            try:
                item["clap_edited_target"] = scorers["clap"].score_audio_text(edt_audio, tgt_prompt)
                item["clap_edited_source"] = scorers["clap"].score_audio_text(edt_audio, src_prompt)
                item["clap_source_source"] = scorers["clap"].score_audio_text(src_audio, src_prompt)
                item["clap_delta_target_minus_source"] = item["clap_edited_target"] - item["clap_edited_source"]
                item["clap_status"] = "ok"
                item["clap_error"] = ""
            except Exception as e:
                item["clap_edited_target"] = np.nan
                item["clap_edited_source"] = np.nan
                item["clap_source_source"] = np.nan
                item["clap_delta_target_minus_source"] = np.nan
                item["clap_status"] = "failed"
                item["clap_error"] = repr(e)

        if "lpaps" in scorers:
            try:
                item["lpaps_source_edited"] = scorers["lpaps"].score_pair(src_audio, edt_audio)
                item["lpaps_num_valid_windows"] = scorers["lpaps"].last_num_windows
                item["lpaps_num_failed_windows"] = scorers["lpaps"].last_num_failed_windows
                item["lpaps_status"] = "ok"
                item["lpaps_error"] = scorers["lpaps"].last_error
            except Exception as e:
                item["lpaps_source_edited"] = np.nan
                item["lpaps_num_valid_windows"] = getattr(scorers.get("lpaps"), "last_num_windows", 0)
                item["lpaps_num_failed_windows"] = getattr(scorers.get("lpaps"), "last_num_failed_windows", 0)
                item["lpaps_status"] = "failed"
                item["lpaps_error"] = repr(e)

        if "cqt" in scorers:
            try:
                item["cqt1_pcc_source_edited"] = scorers["cqt"].score_pair(src_audio, edt_audio)
                item["cqt_status"] = "ok"
                item["cqt_error"] = ""
            except Exception as e:
                item["cqt_status"] = "failed"
                item["cqt_error"] = repr(e)

        if "cdpam" in scorers:
            try:
                item["cdpam_source_edited"] = scorers["cdpam"].score_pair(src_audio, edt_audio)
                item["cdpam_status"] = "ok"
                item["cdpam_error"] = ""
            except Exception as e:
                item["cdpam_status"] = "failed"
                item["cdpam_error"] = repr(e)

        if "detac" in scorers:
            try:
                detac_metrics = scorers["detac"].score_pair(src_audio, edt_audio)
                item.update(detac_metrics)
                item["detac_status"] = "ok"
                item["detac_error"] = ""
            except Exception as e:
                item["detac_status"] = "failed"
                item["detac_error"] = repr(e)

        rows.append(item)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    out = pd.DataFrame(rows)

    fad_results = compute_fad_metrics(out, out_csv, args)
    if fad_results:
        for k, v in fad_results.items():
            out[k] = v

    out.to_csv(out_csv, index=False)
    summarize(out, out_summary)

    print("[INFO] finished")
    print("[INFO] csv:", out_csv)
    print("[INFO] summary:", out_summary)


if __name__ == "__main__":
    main()
