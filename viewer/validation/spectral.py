"""
Cohort-level PSD computation.

Computes per-subject, per-channel Welch PSDs from artifact-free segments,
storing them in a SpectraResult for downstream plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_viewer.io import RecordingKey, load_eeg_dict
from eeg_viewer.artifacts import ArtifactModel
from eeg_viewer.views.psd import welch_psd_numpy

from .scanner import eeg_channel_indices, is_aux_channel


@dataclass
class SpectraResult:
    """Container for cohort-level spectral data."""

    # Frequency axis (shared across all subjects)
    freqs: np.ndarray = field(default_factory=lambda: np.array([]))

    # PSD arrays: list of (n_eeg_channels, n_freqs) per recording
    # Indexed by recording key string "sub_ses_cond"
    psds: Dict[str, np.ndarray] = field(default_factory=dict)

    # Channel labels for each recording (EEG-only subset)
    channel_labels: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata for ordering
    recording_keys: List[str] = field(default_factory=list)
    conditions: Dict[str, str] = field(default_factory=dict)
    subjects: Dict[str, str] = field(default_factory=dict)

    def all_channel_mean_psd(self) -> np.ndarray:
        """
        Return (n_recordings, n_freqs) array: mean PSD across EEG channels
        per recording.
        """
        arrs = []
        for k in self.recording_keys:
            if k in self.psds:
                arrs.append(np.mean(self.psds[k], axis=0))
        if not arrs:
            return np.array([])
        return np.vstack(arrs)

    def by_condition(self, cond: str) -> List[str]:
        """Return recording keys for a given condition."""
        return [k for k in self.recording_keys if self.conditions.get(k) == cond]

    def channel_psd_for_region(
        self, key: str, region_channels: List[str]
    ) -> Optional[np.ndarray]:
        """
        Return mean PSD across channels in region_channels for a given recording.
        Returns None if no channels match.
        """
        if key not in self.psds or key not in self.channel_labels:
            return None
        labels = self.channel_labels[key]
        indices = [i for i, lab in enumerate(labels) if lab in region_channels]
        if not indices:
            return None
        return np.mean(self.psds[key][indices], axis=0)


# ── Region definitions ──────────────────────────────────────────────────

REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8"],
    "central": ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "T7", "T8"],
    "parietal": ["CP3", "CPz", "CP4", "P3", "Pz", "P4", "P7", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}


# ── Main computation ────────────────────────────────────────────────────

def compute_cohort_spectra(
    recordings: Dict[RecordingKey, Dict[str, Path]],
    cohort_df: pd.DataFrame,
    nperseg_s: float = 2.0,
) -> SpectraResult:
    """
    Compute Welch PSD for every recording, using only artifact-free samples.

    Uses the same welch_psd_numpy as the viewer's PSD view.
    """
    result = SpectraResult()

    total = sum(len(v) for v in recordings.values())
    done = 0

    target_freqs: Optional[np.ndarray] = None

    for key, conds in sorted(recordings.items(), key=lambda x: x[0].label()):
        for cond_label, path in sorted(conds.items()):
            done += 1
            rec_key = f"{key.sub}_{key.ses}_{cond_label}"

            try:
                eeg = load_eeg_dict(path)
            except Exception:
                continue

            fs = int(eeg.get("Fs", 1))
            data = np.squeeze(np.asarray(eeg.get("data")))
            if data.ndim == 1:
                data = data[None, :]
            elif data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            if data.ndim != 2:
                continue

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
                continue

            eeg_labels = [labels[i] for i in eeg_idxs]

            # Build clean signal: exclude artifact samples
            if art.artifact_mask is not None:
                clean_mask = ~art.artifact_mask
            else:
                clean_mask = np.ones(n_samp, dtype=bool)

            n_clean = int(clean_mask.sum())
            nperseg = int(min(n_clean, max(256, int(fs * nperseg_s))))
            noverlap = nperseg // 2

            if n_clean < nperseg:
                continue

            # Compute PSD per EEG channel
            channel_psds = []
            valid = True
            for ch_idx in eeg_idxs:
                x = data[ch_idx].astype(np.float64)[clean_mask]
                f, pxx = welch_psd_numpy(x, fs, nperseg=nperseg, noverlap=noverlap)
                if f.size == 0:
                    valid = False
                    break

                # On first successful computation, set target freqs
                if target_freqs is None:
                    target_freqs = f
                    result.freqs = f

                # Ensure frequency alignment via interpolation if needed
                if len(f) != len(target_freqs) or not np.allclose(f, target_freqs):
                    pxx = np.interp(target_freqs, f, pxx)

                channel_psds.append(pxx)

            if not valid or not channel_psds:
                continue

            psd_array = np.vstack(channel_psds)  # (n_eeg_ch, n_freqs)

            result.psds[rec_key] = psd_array
            result.channel_labels[rec_key] = eeg_labels
            result.recording_keys.append(rec_key)
            result.conditions[rec_key] = cond_label
            result.subjects[rec_key] = key.sub

            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] spectra computed")

    print(f"  → {len(result.recording_keys)} spectra computed, "
          f"{len(result.freqs)} frequency bins")
    return result
