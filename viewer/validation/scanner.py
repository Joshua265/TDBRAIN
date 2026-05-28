"""
Batch scanner: iterates over all recordings, extracts per-subject metadata
into a pandas DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_viewer.io import RecordingKey, load_eeg_dict, parse_cond
from eeg_viewer.artifacts import ArtifactModel, mask_to_segments

# ── Channel classification (shared with viewer) ──────────────────────────

_AUX_EXACT = {
    "veog",
    "heog",
    "ecg",
    "erbs",
    "orbocc",
    "mass",
    "artifacts",
    "artifact",
}

# Standard 10-20 EEG channel set from the pipeline
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC3",
    "FCz",
    "FC4",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP3",
    "CPz",
    "CP4",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
]


def is_aux_channel(label: str) -> bool:
    s = label.strip().lower()
    if s in _AUX_EXACT:
        return True
    if "eog" in s or "ecg" in s or "emg" in s or "artifact" in s:
        return True
    return False


def eeg_channel_indices(labels: List[str]) -> List[int]:
    """Return indices of non-auxiliary EEG channels."""
    return [i for i, lab in enumerate(labels) if not is_aux_channel(str(lab))]


# ── Epoch helpers ────────────────────────────────────────────────────────


def count_good_epochs(
    artifact_mask: Optional[np.ndarray],
    n_samples: int,
    fs: int,
    epoch_len_s: float = 4.0,
    overlap: float = 0.5,
) -> Tuple[int, int]:
    """
    Count total and good (artifact-free) epochs using the supplement's
    4-second windows with 50% overlap.

    Returns (n_total_epochs, n_good_epochs).
    """
    epoch_samp = int(512)
    step_samp = int(epoch_samp * (1.0 - 0))
    if step_samp <= 0:
        step_samp = epoch_samp

    n_total = 0
    n_good = 0

    for start in range(0, n_samples - epoch_samp + 1, step_samp):
        n_total += 1
        if artifact_mask is not None:
            epoch_mask = artifact_mask[start : start + epoch_samp]
            if not np.any(epoch_mask):
                n_good += 1
        else:
            n_good += 1

    return n_total, n_good


def per_channel_artifact_burden(
    data: np.ndarray,
    labels: List[str],
    artifact_model: ArtifactModel,
    fs: int,
    epoch_len_s: float = 4.0,
    overlap: float = 0.5,
) -> Dict[str, float]:
    """
    Compute per-channel artifact burden as fraction of epochs that are bad.

    Uses the per-channel sample masks from the artifact detectors
    (EMGsamps, JUMPsamps, etc.) collapsed into a single per-channel mask.
    Falls back to the global artifact mask if per-channel data is not available.
    """
    n_ch, n_samp = data.shape
    epoch_samp = int(epoch_len_s * fs)
    step_samp = int(epoch_samp * (1.0 - overlap))
    if step_samp <= 0:
        step_samp = epoch_samp

    result: Dict[str, float] = {}
    eeg_idxs = eeg_channel_indices(labels)

    # Build per-channel artifact mask from sample masks
    ch_masks = np.zeros((n_ch, n_samp), dtype=bool)
    for _k, arr in artifact_model.sample_masks.items():
        if arr.shape[0] == n_ch and arr.shape[1] == n_samp:
            ch_masks |= arr != 0
        elif arr.shape[0] <= n_ch:
            ch_masks[: arr.shape[0]] |= arr[:, :n_samp] != 0

    # If no per-channel masks, fall back to global mask
    if not artifact_model.sample_masks and artifact_model.artifact_mask is not None:
        for i in eeg_idxs:
            ch_masks[i] = artifact_model.artifact_mask[:n_samp]

    for i in eeg_idxs:
        if i >= len(labels):
            continue
        lab = str(labels[i])
        n_total = 0
        n_bad = 0
        for start in range(0, n_samp - epoch_samp + 1, step_samp):
            n_total += 1
            if np.any(ch_masks[i, start : start + epoch_samp]):
                n_bad += 1
        result[lab] = n_bad / max(n_total, 1)

    return result


# ── Main scanner ─────────────────────────────────────────────────────────


def _process_one(
    path: Path,
    tsv_df: Optional[pd.DataFrame],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, float]]]:
    """
    Load a single .npy recording and extract all metadata.

    Returns (row_dict, per_channel_dict) or (None, None) on error.
    """
    try:
        eeg = load_eeg_dict(path)
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return None, None

    fs = int(eeg.get("Fs", 1))
    data = np.squeeze(np.asarray(eeg.get("data")))
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        print(f"  [WARN] Unexpected shape {data.shape} in {path.name}")
        return None, None

    n_ch, n_samp = data.shape

    labels = eeg.get("labels")
    if isinstance(labels, np.ndarray):
        labels = [str(x) for x in labels.tolist()]
    elif isinstance(labels, list):
        labels = [str(x) for x in labels]
    else:
        labels = [f"ch{i}" for i in range(n_ch)]

    art = ArtifactModel(eeg, data, labels, fs)

    # Basic info
    cond = parse_cond(path.name) or "?"
    from eeg_viewer.io import parse_sub, parse_ses

    sub = parse_sub(path.name) or "?"
    ses = parse_ses(str(path))
    duration_s = n_samp / float(fs)

    eeg_idxs = eeg_channel_indices(labels)
    n_eeg = len(eeg_idxs)

    # Artifact coverage
    cov = art.artifact_coverage_percent()
    n_total_epochs, n_good_epochs = count_good_epochs(art.artifact_mask, n_samp, fs)

    # Per-channel burden
    ch_burden = per_channel_artifact_burden(data, labels, art, fs)
    n_good_channels = sum(1 for v in ch_burden.values() if v < 1.0 / 3.0)

    # Detector counts
    det_counts: Dict[str, int] = {}
    for k, segs in art.detector_segments.items():
        det_counts[k] = segs.shape[0] if segs.size > 0 else 0

    row: Dict[str, Any] = {
        "sub": sub,
        "ses": ses,
        "condition": cond,
        "path": str(path),
        "fs": fs,
        "n_channels": n_ch,
        "n_eeg_channels": n_eeg,
        "n_samples": n_samp,
        "duration_s": duration_s,
        "has_artifact_mask": art.has_global_mask(),
        "artifact_coverage_pct": cov if cov is not None else np.nan,
        "n_total_epochs": n_total_epochs,
        "n_good_epochs": n_good_epochs,
        "pct_good_epochs": 100.0 * n_good_epochs / max(n_total_epochs, 1),
        "n_good_channels": n_good_channels,
    }

    # Add individual detector counts
    for k, cnt in det_counts.items():
        row[f"det_{k}"] = cnt

    # Demographics from TSV if available
    if tsv_df is not None and "participants_ID" in tsv_df.columns:
        match = tsv_df[tsv_df["participants_ID"].astype(str) == sub]
        if not match.empty:
            r0 = match.iloc[0]
            for col in tsv_df.columns:
                if col != "participants_ID":
                    row[f"tsv_{col}"] = r0.get(col, np.nan)

    return row, ch_burden


def scan_cohort(
    recordings: Dict[RecordingKey, Dict[str, Path]],
    tsv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Scan all recordings and return:
    - cohort_df: one row per recording with metadata
    - per_channel_artifact: dict keyed by file path string → {channel: burden}
    """
    tsv_df = None
    if tsv_path and tsv_path.exists():
        tsv_df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    rows: List[Dict[str, Any]] = []
    per_channel: Dict[str, Dict[str, float]] = {}

    total = sum(len(v) for v in recordings.values())
    done = 0

    for key, conds in sorted(recordings.items(), key=lambda x: x[0].label()):
        for cond_label, path in sorted(conds.items()):
            done += 1
            print(f"  [{done}/{total}] {path.name}")
            row, ch_burden = _process_one(path, tsv_df)
            if row is not None:
                rows.append(row)
            if ch_burden is not None:
                per_channel[str(path)] = ch_burden

    df = pd.DataFrame(rows)
    return df, per_channel
