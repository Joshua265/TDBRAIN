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

NUM_CHANNELS = 26

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


def _collapse_per_channel_sample_masks(
    artifact_model: ArtifactModel,
    n_ch: int,
    n_samp: int,
) -> Optional[np.ndarray]:
    """Collapse detector sample masks into a single (samples, channels) mask.

    Returns None if no per-channel masks are available.
    """

    if not artifact_model.sample_masks:
        return None

    ch_masks = np.zeros((n_samp, n_ch), dtype=bool)
    for _k, arr in artifact_model.sample_masks.items():
        a = np.asarray(arr)

        # Expected shape in this project is usually (channels, samples).
        if a.ndim != 2:
            continue

        if a.shape[0] == n_ch and a.shape[1] >= 1:
            # (channels, samples)
            ch_masks[:, : min(n_ch, a.shape[0])] |= (a[:n_ch, :n_samp].T != 0)
        elif a.shape[1] == n_ch and a.shape[0] >= 1:
            # (samples, channels)
            ch_masks[:, : min(n_ch, a.shape[1])] |= (a[:n_samp, :n_ch] != 0)

    return ch_masks


def _extract_tdbrain_data_quality_flag(eeg_dict: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of TDBRAIN pipeline quality flag."""

    for key in [
        "data quality",
        "data_quality",
        "dataQuality",
        "tdbrain_data_quality",
        "tdbrain quality",
        "quality",
    ]:
        if key in eeg_dict:
            val = eeg_dict.get(key)
            return None if val is None else str(val)

    info = eeg_dict.get("info")
    if isinstance(info, dict):
        for key in ["data quality", "data_quality", "quality"]:
            if key in info:
                val = info.get(key)
                return None if val is None else str(val)

    return None


def _check_tdbrain_quality(
    artifact_mask: np.ndarray,
    eeg_data: np.ndarray,
    sampling_rate: int,
    epoch_seconds: float = 4.0,
    epoch_overlap: float = 0.5,
    pct_good_epochs_threshold: float = 50.0,
    n_good_channels_threshold: int = 20,
    num_channels: int = NUM_CHANNELS,
    bad_channels_per_epoch_threshold: int = 3,
    bad_channel_epoch_fraction_threshold: float = 1.0 / 3.0,
    tdbrain_data_quality: Optional[str] = None,
    count_whole_epoch_rejection_for_channel: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Check TDBRAIN-style dataset-level quality criteria.

    Implements three layers:

    1. TDBRAIN pipeline flag: reject if `tdbrain_data_quality == "bad"`.
    2. Epoch-level artifact QC:
       - With a per-channel mask, reject an epoch if >=3 EEG channels are bad.
       - With only a global artifacts channel, reject an epoch if any artifact is
         present (approximation).
    3. Channel-level QC:
       - With a per-channel mask, reject a channel if >33% of epochs are bad.
       - With only a global mask, fall back to signal-validity checks.

    Returns (passed, metrics).
    """

    metrics: Dict[str, Any] = {}

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be > 0")

    if epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be > 0")

    if not (0.0 <= epoch_overlap < 1.0):
        raise ValueError("epoch_overlap must be in [0, 1)")

    eeg_data = np.asarray(eeg_data)

    if eeg_data.ndim != 2:
        raise ValueError("eeg_data must be 2D with shape (samples, channels)")

    n_eeg_samples, n_available_channels = eeg_data.shape
    n_eval_channels = min(num_channels, n_available_channels)

    if n_eval_channels <= 0:
        raise ValueError("No EEG channels available for QC")

    tdbrain_quality_bad = False
    if tdbrain_data_quality is not None:
        tdbrain_quality_bad = str(tdbrain_data_quality).strip().lower() == "bad"

    metrics["tdbrain_data_quality"] = tdbrain_data_quality
    metrics["tdbrain_quality_bad"] = tdbrain_quality_bad

    mask = np.asarray(artifact_mask)
    mask = np.squeeze(mask)
    mask = np.nan_to_num(mask, nan=0.0) != 0

    artifact_mode: str

    if mask.ndim == 1:
        global_sample_mask = mask
        channel_sample_mask = None
        artifact_mode = "global"

    elif mask.ndim == 2:
        if mask.shape[0] == n_eeg_samples:
            channel_sample_mask = mask[:, :n_eval_channels]
        elif mask.shape[1] == n_eeg_samples:
            channel_sample_mask = mask[:n_eval_channels, :].T
        else:
            raise ValueError(
                "Could not align 2D artifact_mask with eeg_data. Expected "
                "(samples, channels) or (channels, samples). Got "
                f"artifact_mask.shape={mask.shape}, eeg_data.shape={eeg_data.shape}."
            )

        global_sample_mask = None
        artifact_mode = "per_channel"

    else:
        raise ValueError(
            "artifact_mask must be 1D global mask or 2D per-channel mask. "
            f"Got shape {artifact_mask.shape} after squeeze -> {mask.shape}."
        )

    metrics["artifact_mask_mode"] = artifact_mode
    metrics["has_exact_channel_artifact_qc"] = artifact_mode == "per_channel"

    if artifact_mode == "global":
        assert global_sample_mask is not None
        n_mask_samples = int(global_sample_mask.shape[0])
    else:
        assert channel_sample_mask is not None
        n_mask_samples = int(channel_sample_mask.shape[0])

    total_samples = int(min(n_eeg_samples, n_mask_samples))

    metrics["n_eeg_samples"] = int(n_eeg_samples)
    metrics["n_artifact_mask_samples"] = int(n_mask_samples)
    metrics["n_samples_used"] = int(total_samples)
    metrics["n_eval_channels"] = int(n_eval_channels)

    epoch_samples = int(round(epoch_seconds * sampling_rate))
    step_samples = int(round(epoch_samples * (1.0 - epoch_overlap)))
    step_samples = max(1, step_samples)

    metrics["epoch_seconds"] = float(epoch_seconds)
    metrics["epoch_samples"] = int(epoch_samples)
    metrics["epoch_overlap"] = float(epoch_overlap)
    metrics["step_samples"] = int(step_samples)

    if total_samples < epoch_samples:
        n_epochs = 0
        n_bad_epochs = 0
        n_good_epochs = 0
        pct_good_epochs = 0.0
        pct_bad_epochs = 100.0
        bad_epoch = np.array([], dtype=bool)
        epoch_channel_bad = None
    else:
        epoch_starts = np.arange(
            0,
            total_samples - epoch_samples + 1,
            step_samples,
            dtype=int,
        )

        n_epochs = len(epoch_starts)

        if artifact_mode == "per_channel":
            assert channel_sample_mask is not None
            epoch_channel_bad = np.zeros((n_epochs, n_eval_channels), dtype=bool)

            for i, start in enumerate(epoch_starts):
                end = start + epoch_samples
                epoch_channel_bad[i, :] = channel_sample_mask[start:end, :].any(axis=0)

            n_bad_channels_per_epoch = epoch_channel_bad.sum(axis=1)
            bad_epoch = n_bad_channels_per_epoch >= bad_channels_per_epoch_threshold

            metrics["bad_channels_per_epoch_min"] = int(n_bad_channels_per_epoch.min())
            metrics["bad_channels_per_epoch_max"] = int(n_bad_channels_per_epoch.max())
            metrics["bad_channels_per_epoch_mean"] = float(n_bad_channels_per_epoch.mean())

        else:
            assert global_sample_mask is not None
            epoch_channel_bad = None
            bad_epoch = np.zeros(n_epochs, dtype=bool)

            for i, start in enumerate(epoch_starts):
                end = start + epoch_samples
                bad_epoch[i] = global_sample_mask[start:end].any()

            metrics["global_artifact_epoch_rule"] = "any artifact sample in epoch"
            metrics["warning_global_mask"] = (
                "Only a global artifact mask was provided. "
                "Per-channel >33% bad-channel QC cannot be reproduced exactly."
            )

        n_bad_epochs = int(bad_epoch.sum())
        n_good_epochs = int(n_epochs - n_bad_epochs)
        pct_good_epochs = (n_good_epochs / n_epochs) * 100.0 if n_epochs else 0.0
        pct_bad_epochs = (n_bad_epochs / n_epochs) * 100.0 if n_epochs else 100.0

    metrics["n_epochs"] = int(n_epochs)
    metrics["n_good_epochs"] = int(n_good_epochs)
    metrics["n_bad_epochs"] = int(n_bad_epochs)
    metrics["pct_good_epochs"] = float(pct_good_epochs)
    metrics["pct_bad_epochs"] = float(pct_bad_epochs)

    valid_samples = total_samples

    signal_good_channel = np.zeros(n_eval_channels, dtype=bool)
    for ch in range(n_eval_channels):
        ch_data = eeg_data[:valid_samples, ch]
        signal_good_channel[ch] = (
            np.all(np.isfinite(ch_data)) and np.nanvar(ch_data) > 0
        )

    artifact_bad_channel = np.zeros(n_eval_channels, dtype=bool)
    bad_epoch_fraction_per_channel = np.zeros(n_eval_channels, dtype=float)

    if artifact_mode == "per_channel" and n_epochs > 0:
        assert epoch_channel_bad is not None
        channel_epoch_bad_for_fraction = epoch_channel_bad.copy()

        if count_whole_epoch_rejection_for_channel:
            channel_epoch_bad_for_fraction[bad_epoch, :] = True

        bad_epoch_fraction_per_channel = channel_epoch_bad_for_fraction.mean(axis=0)

        artifact_bad_channel = (
            bad_epoch_fraction_per_channel > bad_channel_epoch_fraction_threshold
        )

    good_channel = signal_good_channel & ~artifact_bad_channel

    metrics["n_signal_bad_channels"] = int((~signal_good_channel).sum())
    metrics["n_artifact_bad_channels"] = int(artifact_bad_channel.sum())
    metrics["n_good_channels"] = int(good_channel.sum())
    metrics["bad_channel_epoch_fraction_threshold"] = float(
        bad_channel_epoch_fraction_threshold
    )
    metrics["bad_channel_indices"] = np.where(~good_channel)[0].tolist()
    metrics["artifact_bad_channel_indices"] = np.where(artifact_bad_channel)[0].tolist()
    metrics["signal_bad_channel_indices"] = np.where(~signal_good_channel)[0].tolist()
    metrics["bad_epoch_fraction_per_channel"] = bad_epoch_fraction_per_channel.tolist()

    reject_bad_epochs = pct_good_epochs < pct_good_epochs_threshold
    reject_too_few_channels = int(metrics["n_good_channels"]) < n_good_channels_threshold
    reject_no_epochs = n_epochs == 0

    passed = not (
        tdbrain_quality_bad
        or reject_bad_epochs
        or reject_too_few_channels
        or reject_no_epochs
    )

    metrics["reject_tdbrain_quality_bad"] = bool(tdbrain_quality_bad)
    metrics["reject_bad_epochs"] = bool(reject_bad_epochs)
    metrics["reject_too_few_channels"] = bool(reject_too_few_channels)
    metrics["reject_no_epochs"] = bool(reject_no_epochs)
    metrics["pct_good_epochs_threshold"] = float(pct_good_epochs_threshold)
    metrics["n_good_channels_threshold"] = int(n_good_channels_threshold)
    metrics["passed"] = bool(passed)

    return bool(passed), metrics


# ── Epoch helpers ────────────────────────────────────────────────────────


def count_good_epochs(
    artifact_mask: Optional[np.ndarray],
    n_samples: int,
    fs: int,
    epoch_len_s: float = 4.0,
    overlap: float = 0.5,
) -> Tuple[int, int]:
    """Compatibility helper for legacy callers.

    Uses the same epoching logic as `_check_tdbrain_quality` with a *global* mask:
    an epoch is good if it contains no artifact samples.

    Returns (n_total_epochs, n_good_epochs).
    """

    if artifact_mask is None:
        artifact_mask_arr = np.zeros(int(n_samples), dtype=bool)
    else:
        artifact_mask_arr = np.asarray(artifact_mask)[: int(n_samples)]

    # Dummy single-channel EEG to satisfy shape requirements.
    eeg_dummy = np.zeros((int(n_samples), 1), dtype=float)
    passed, metrics = _check_tdbrain_quality(
        artifact_mask=artifact_mask_arr,
        eeg_data=eeg_dummy,
        sampling_rate=int(fs),
        epoch_seconds=float(epoch_len_s),
        epoch_overlap=float(overlap),
        pct_good_epochs_threshold=0.0,
        n_good_channels_threshold=0,
        num_channels=1,
        tdbrain_data_quality=None,
    )
    _ = passed

    return int(metrics["n_epochs"]), int(metrics["n_good_epochs"])


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

    tdbrain_data_quality = _extract_tdbrain_data_quality_flag(eeg)

    if n_eeg == 0:
        print(f"  [WARN] No EEG channels detected in {path.name}")
        qc_passed = False
        qc_metrics = {
            "artifact_mask_mode": None,
            "n_epochs": 0,
            "n_good_epochs": 0,
            "pct_good_epochs": 0.0,
            "n_good_channels": 0,
            "reject_tdbrain_quality_bad": False,
            "reject_bad_epochs": False,
            "reject_too_few_channels": True,
            "reject_no_epochs": True,
        }
        ch_burden: Dict[str, float] = {}
    else:
        eeg_data = data[eeg_idxs].T  # (samples, eeg_channels)
        n_eval_channels = min(NUM_CHANNELS, eeg_data.shape[1])

        per_ch_mask_all = _collapse_per_channel_sample_masks(art, n_ch, n_samp)
        if per_ch_mask_all is not None:
            artifact_mask = per_ch_mask_all[:n_samp, :][:, eeg_idxs]
        elif art.artifact_mask is not None:
            artifact_mask = np.asarray(art.artifact_mask)[:n_samp]
        else:
            artifact_mask = np.zeros(int(n_samp), dtype=bool)

        qc_passed, qc_metrics = _check_tdbrain_quality(
            artifact_mask=artifact_mask,
            eeg_data=eeg_data,
            sampling_rate=int(fs),
            epoch_seconds=4.0,
            epoch_overlap=0.5,
            tdbrain_data_quality=tdbrain_data_quality,
            num_channels=int(n_eval_channels),
        )

        # Per-channel burden (fraction of bad epochs per channel)
        ch_burden = per_channel_artifact_burden(data, labels, art, fs)

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
        "tdbrain_data_quality": tdbrain_data_quality,
        "artifact_mask_mode": qc_metrics.get("artifact_mask_mode"),
        "n_total_epochs": int(qc_metrics["n_epochs"]),
        "n_good_epochs": int(qc_metrics["n_good_epochs"]),
        "pct_good_epochs": float(qc_metrics["pct_good_epochs"]),
        "n_good_channels": int(qc_metrics["n_good_channels"]),
        "qc_passed": bool(qc_passed),
        "reject_tdbrain_quality_bad": bool(qc_metrics.get("reject_tdbrain_quality_bad", False)),
        "reject_bad_epochs": bool(qc_metrics.get("reject_bad_epochs", False)),
        "reject_too_few_channels": bool(qc_metrics.get("reject_too_few_channels", False)),
        "reject_no_epochs": bool(qc_metrics.get("reject_no_epochs", False)),
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
