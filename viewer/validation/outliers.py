"""
Spectral outlier detection — must-have 5.

Computes multiple outlier scores per subject, ranks them, and returns
a summary DataFrame.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .spectral import SpectraResult


# ── Band definitions ─────────────────────────────────────────────────────

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _band_power(freqs: np.ndarray, psd: np.ndarray, flo: float, fhi: float) -> float:
    """Integrate PSD between flo and fhi (trapezoidal)."""
    mask = (freqs >= flo) & (freqs <= fhi)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation."""
    return float(np.median(np.abs(x - np.median(x))))


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using MAD."""
    med = np.median(x)
    m = _mad(x)
    if m < 1e-15:
        return np.zeros_like(x)
    return (x - med) / (m * 1.4826)  # 1.4826 makes MAD consistent with std for normal


# ── Outlier scoring ─────────────────────────────────────────────────────

def compute_outlier_scores(
    spectra: SpectraResult,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-recording outlier scores and return a ranked DataFrame.

    Scores:
    1. Robust spectral distance (MAD distance from cohort median)
    2. Frequency-wise Tukey outlier count
    3. Bandpower MAD-z scores (delta, theta, alpha, beta, gamma)
    4. Line-noise score (50 Hz relative to neighbors)
    5. Low-frequency artifact score (excess 1–5 Hz)
    6. High-frequency slope (30–45 Hz)
    """
    if not spectra.recording_keys:
        return pd.DataFrame()

    freqs = spectra.freqs
    conditions = sorted(set(spectra.conditions.values()))

    rows = []

    for cond in conditions:
        keys = spectra.by_condition(cond)
        if len(keys) < 3:
            continue

        # Build mean-PSD matrix (n_subjects × n_freqs)
        mean_psds = []
        for k in keys:
            mean_psds.append(np.mean(spectra.psds[k], axis=0))
        mat = np.vstack(mean_psds)
        mat_db = 10.0 * np.log10(np.maximum(mat, 1e-20))

        # Cohort median spectrum (dB)
        cohort_median = np.median(mat_db, axis=0)
        cohort_mad = np.array([_mad(mat_db[:, j]) for j in range(mat_db.shape[1])])

        # Tukey fences per frequency
        q25 = np.percentile(mat_db, 25, axis=0)
        q75 = np.percentile(mat_db, 75, axis=0)
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr

        # Bandpower arrays
        bp: Dict[str, np.ndarray] = {}
        for band_name, (flo, fhi) in BANDS.items():
            bp[band_name] = np.array([_band_power(freqs, mat[i], flo, fhi) for i in range(len(keys))])

        # Line noise indices (50 Hz ± neighbors)
        line_freq = 50.0
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        line_idx = np.argmin(np.abs(freqs - line_freq))
        neighbor_lo = max(0, line_idx - int(3 / df))
        neighbor_hi = min(len(freqs) - 1, line_idx + int(3 / df))

        for i, k in enumerate(keys):
            sub = spectra.subjects.get(k, "?")

            # 1. Robust spectral distance
            diff = mat_db[i] - cohort_median
            safe_mad = np.where(cohort_mad > 1e-10, cohort_mad, 1.0)
            spectral_dist = float(np.median(np.abs(diff) / (safe_mad * 1.4826)))

            # 2. Tukey outlier count
            below = mat_db[i] < lower_fence
            above = mat_db[i] > upper_fence
            tukey_count = int(below.sum() + above.sum())

            # 3. Bandpower z-scores
            bp_scores: Dict[str, float] = {}
            for band_name, arr in bp.items():
                z = _robust_z(arr)
                bp_scores[f"z_{band_name}"] = float(z[i])

            # 4. Line noise score
            if line_idx > 0 and line_idx < len(freqs) - 1:
                neighbor_power = np.mean(
                    np.concatenate([
                        mat_db[i, neighbor_lo:line_idx],
                        mat_db[i, line_idx + 1:neighbor_hi + 1],
                    ])
                )
                line_score = float(mat_db[i, line_idx] - neighbor_power)
            else:
                line_score = 0.0

            # 5. Low-frequency artifact (1–5 Hz excess)
            lf_mask = (freqs >= 1.0) & (freqs <= 5.0)
            if lf_mask.sum() > 0:
                lf_excess = float(np.mean(mat_db[i, lf_mask] - cohort_median[lf_mask]))
            else:
                lf_excess = 0.0

            # 6. High-frequency slope (30–45 Hz)
            hf_mask = (freqs >= 30.0) & (freqs <= 45.0)
            if hf_mask.sum() > 0:
                hf_excess = float(np.mean(mat_db[i, hf_mask] - cohort_median[hf_mask]))
            else:
                hf_excess = 0.0

            # Composite outlier score (weighted sum of absolute scores)
            composite = (
                spectral_dist * 2.0
                + tukey_count / max(len(freqs), 1) * 10.0
                + abs(line_score) * 0.5
                + abs(lf_excess) * 1.0
                + abs(hf_excess) * 1.0
            )

            # Determine main reason
            reasons = []
            if lf_excess > 3.0:
                reasons.append("excessive 1–5 Hz power")
            if line_score > 5.0:
                reasons.append("strong 50 Hz residual")
            if hf_excess > 3.0:
                reasons.append("broad high-frequency power (possible EMG)")
            if tukey_count > len(freqs) * 0.3:
                reasons.append("many Tukey outlier bins")
            if not reasons:
                if spectral_dist > 3.0:
                    reasons.append("high overall spectral distance")
                else:
                    reasons.append("within normal range")

            row = {
                "sub": sub,
                "condition": cond,
                "rec_key": k,
                "composite_outlier_score": round(composite, 3),
                "spectral_distance": round(spectral_dist, 3),
                "tukey_outlier_bins": tukey_count,
                "line_noise_score_dB": round(line_score, 2),
                "lf_excess_dB": round(lf_excess, 2),
                "hf_excess_dB": round(hf_excess, 2),
                "main_reason": "; ".join(reasons),
            }
            row.update({k2: round(v2, 3) for k2, v2 in bp_scores.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("composite_outlier_score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    return df
