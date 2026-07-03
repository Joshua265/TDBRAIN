"""
Cohort-level channel connectivity computation.

Computes per-recording, band-averaged coherence matrices on artifact-free
samples, using the implementation in coherence.py, parallelized over files.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_viewer.artifacts import ArtifactModel
from eeg_viewer.io import RecordingKey, load_eeg_dict

from validation.coherence import compute_coherence_matrices_by_band

from .scanner import eeg_channel_indices


@dataclass
class CoherenceResult:
    band_mats: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    # band_mats[rec_key][band_name] -> (n_ch, n_ch)

    channel_labels: Dict[str, List[str]] = field(default_factory=dict)
    subjects: Dict[str, str] = field(default_factory=dict)
    conditions: Dict[str, str] = field(default_factory=dict)
    recording_keys: List[str] = field(default_factory=list)


def _process_one(
    path: Path,
    rec_key: str,
    sub: str,
    cond: str,
    bands: List[Tuple[str, float, float]],
    nperseg: int,
) -> Optional[Tuple[str, Dict[str, np.ndarray], List[str], str, str]]:
    try:
        eeg = load_eeg_dict(path)
    except Exception:
        return None

    fs = int(eeg.get("Fs", 1))
    data = np.squeeze(np.asarray(eeg.get("data")))
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        return None

    n_ch, n_samp = data.shape

    labels = eeg.get("labels")
    if isinstance(labels, np.ndarray):
        labels = [str(x) for x in labels.tolist()]
    elif isinstance(labels, list):
        labels = [str(x) for x in labels]
    else:
        labels = [f"ch{i}" for i in range(n_ch)]

    art = ArtifactModel(eeg, data, labels, fs)
    eeg_idxs = eeg_channel_indices(labels)
    if not eeg_idxs:
        return None

    eeg_labels = [labels[i] for i in eeg_idxs]

    if art.artifact_mask is not None:
        clean_mask = ~art.artifact_mask
    else:
        clean_mask = np.ones(n_samp, dtype=bool)

    n_clean = int(clean_mask.sum())
    if n_clean < max(256, nperseg // 2):
        return None

    # Shape expected by coherence code: (samples, channels)
    x = data[eeg_idxs].astype(np.float64)[:, clean_mask].T

    try:
        band_mats = compute_coherence_matrices_by_band(
            x,
            sampling_rate=fs,
            band_freqs=bands,
            nperseg=min(nperseg, x.shape[0]),
        )
    except Exception:
        return None

    return rec_key, band_mats, eeg_labels, sub, cond


def compute_cohort_coherence(
    recordings: Dict[RecordingKey, Dict[str, Path]],
    cohort_df: pd.DataFrame,
    bands: List[Tuple[str, float, float]],
    nperseg: int = 512,
    max_workers: Optional[int] = None,
) -> CoherenceResult:
    """Compute band-averaged coherence matrices for all recordings.

    Parallelized with a process pool — each recording is processed
    independently, which makes it embarrassingly parallel.

    Parameters
    ----------
    recordings:
        Mapping from (sub,ses) -> {cond: path}
    cohort_df:
        Filtered cohort metadata (included recordings only).
    bands:
        List of (name, fmin, fmax) in Hz.
    nperseg:
        Welch segment length passed to coherence.
    max_workers:
        Number of worker processes.  None => os.cpu_count().
    """

    result = CoherenceResult()

    # Build task list
    tasks: List[Tuple[Path, str, str, str, List[Tuple[str, float, float]], int]] = []
    for key, conds in sorted(recordings.items(), key=lambda x: x[0].label()):
        for cond_label, path in sorted(conds.items()):
            rec_key = f"{key.sub}_{key.ses}_{cond_label}"
            tasks.append((path, rec_key, key.sub, cond_label, bands, nperseg))

    total = len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_process_one, *t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            done += 1
            out = fut.result()
            if out is not None:
                rec_key, band_mats, eeg_labels, sub, cond = out
                result.band_mats[rec_key] = band_mats
                result.channel_labels[rec_key] = eeg_labels
                result.subjects[rec_key] = sub
                result.conditions[rec_key] = cond
                result.recording_keys.append(rec_key)

            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] coherence computed")

    print(f"  -> {len(result.recording_keys)} coherence matrices computed")
    return result
