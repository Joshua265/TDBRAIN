"""
Alpha validation plots — must-have 6.

- Alpha power topography (8–13 Hz)
- Individual alpha peak frequency (iAPF) distribution
- F4–F3 alpha asymmetry histogram
- EC vs EO alpha suppression
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .spectral import SpectraResult
from .outliers import _band_power
from .topoplot import plot_topomap, STANDARD_CHANNELS


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Alpha helpers ────────────────────────────────────────────────────────

ALPHA_BAND = (8.0, 13.0)


def _find_iapf(freqs: np.ndarray, psd: np.ndarray, flo: float = 7.0, fhi: float = 13.0) -> Optional[float]:
    """
    Find individual alpha peak frequency in the psd.
    Returns None if no clear peak found.
    """
    mask = (freqs >= flo) & (freqs <= fhi)
    if mask.sum() < 3:
        return None
    f_sub = freqs[mask]
    p_sub = psd[mask]

    # Find peaks in the alpha range
    peaks, props = find_peaks(p_sub, height=np.max(p_sub) * 0.4)
    if len(peaks) == 0:
        return None

    # Return the frequency of the highest peak
    best = peaks[np.argmax(props["peak_heights"])]
    return float(f_sub[best])


def _alpha_asymmetry(
    f4_alpha: float, f3_alpha: float
) -> float:
    """Compute frontal alpha asymmetry: ln(F4) - ln(F3)."""
    return float(np.log(max(f4_alpha, 1e-20)) - np.log(max(f3_alpha, 1e-20)))


# ── Plots ────────────────────────────────────────────────────────────────

def _alpha_topography(spectra: SpectraResult, out: Path) -> None:
    """
    Alpha power (8–13 Hz) scalp map, separate for EC and EO.
    Uses the cohort median alpha power per channel.
    """
    conditions = sorted(set(spectra.conditions.values()))

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(5 * len(conditions), 5), squeeze=False)

    csv_rows = []

    for ci, cond in enumerate(conditions):
        ax = axes[0, ci]
        keys = spectra.by_condition(cond)
        if not keys:
            ax.set_title(f"Alpha — {cond} (no data)")
            continue

        # Compute per-channel alpha power (median across subjects)
        channel_alpha: Dict[str, List[float]] = {}
        for k in keys:
            labels = spectra.channel_labels.get(k, [])
            psd = spectra.psds.get(k)
            if psd is None:
                continue
            for j, lab in enumerate(labels):
                if lab in STANDARD_CHANNELS:
                    bp = _band_power(spectra.freqs, psd[j], *ALPHA_BAND)
                    channel_alpha.setdefault(lab, []).append(bp)

        median_alpha = {ch: float(np.median(vals)) for ch, vals in channel_alpha.items()}

        # Convert to dB for visualization
        median_alpha_db = {ch: 10 * np.log10(max(v, 1e-20)) for ch, v in median_alpha.items()}

        plot_topomap(median_alpha_db, ax, title=f"Alpha Power — {cond}")

        for ch, v in median_alpha.items():
            csv_rows.append({"condition": cond, "channel": ch,
                             "alpha_power": v, "alpha_dB": median_alpha_db.get(ch, np.nan)})

    fig.suptitle("Alpha (8–13 Hz) Power Topography", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out, "13_alpha_topography")
    pd.DataFrame(csv_rows).to_csv(out / "13_alpha_topography.csv", index=False)


def _iapf_distribution(spectra: SpectraResult, out: Path) -> None:
    """
    Histogram of individual alpha peak frequency (iAPF).
    """
    conditions = sorted(set(spectra.conditions.values()))

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(6 * len(conditions), 5), squeeze=False)

    csv_rows = []

    for ci, cond in enumerate(conditions):
        ax = axes[0, ci]
        keys = spectra.by_condition(cond)
        iapfs = []

        for k in keys:
            # Use Pz or average posterior for iAPF detection
            psd = spectra.psds.get(k)
            labels = spectra.channel_labels.get(k, [])
            if psd is None:
                continue

            # Prefer Pz, fallback to posterior average
            pz_idx = None
            posterior = ["Pz", "P3", "P4", "O1", "Oz", "O2"]
            for target in posterior:
                if target in labels:
                    pz_idx = labels.index(target)
                    break

            if pz_idx is not None:
                use_psd = psd[pz_idx]
            else:
                # Average all EEG channels
                use_psd = np.mean(psd, axis=0)

            iapf = _find_iapf(spectra.freqs, use_psd)
            sub = spectra.subjects.get(k, "?")
            csv_rows.append({"sub": sub, "condition": cond, "iAPF_Hz": iapf})
            if iapf is not None:
                iapfs.append(iapf)

        if iapfs:
            ax.hist(iapfs, bins=np.arange(6.5, 14.0, 0.5), color="#9B59B6",
                    edgecolor="white", alpha=0.85)
            ax.axvline(np.median(iapfs), color="#E04040", ls="--", lw=2,
                       label=f"Median: {np.median(iapfs):.1f} Hz")
            ax.legend(fontsize=9)

        ax.set_xlabel("iAPF (Hz)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Individual Alpha Peak — {cond}  (n={len(iapfs)})",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out, "14_iapf_distribution")
    pd.DataFrame(csv_rows).to_csv(out / "14_iapf_distribution.csv", index=False)


def _alpha_asymmetry_histogram(spectra: SpectraResult, out: Path) -> None:
    """
    F4–F3 frontal alpha asymmetry histogram.
    Asymmetry = ln(alpha_F4) - ln(alpha_F3)
    """
    conditions = sorted(set(spectra.conditions.values()))

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(6 * len(conditions), 5), squeeze=False)

    csv_rows = []

    for ci, cond in enumerate(conditions):
        ax = axes[0, ci]
        keys = spectra.by_condition(cond)
        asym_vals = []

        for k in keys:
            psd = spectra.psds.get(k)
            labels = spectra.channel_labels.get(k, [])
            if psd is None:
                continue

            if "F4" not in labels or "F3" not in labels:
                continue

            f4_idx = labels.index("F4")
            f3_idx = labels.index("F3")

            f4_alpha = _band_power(spectra.freqs, psd[f4_idx], *ALPHA_BAND)
            f3_alpha = _band_power(spectra.freqs, psd[f3_idx], *ALPHA_BAND)

            asym = _alpha_asymmetry(f4_alpha, f3_alpha)
            sub = spectra.subjects.get(k, "?")
            asym_vals.append(asym)
            csv_rows.append({
                "sub": sub, "condition": cond,
                "F4_alpha": f4_alpha, "F3_alpha": f3_alpha,
                "asymmetry_ln_F4_minus_ln_F3": asym,
            })

        if asym_vals:
            ax.hist(asym_vals, bins=30, color="#2C5F8A", edgecolor="white", alpha=0.85)
            ax.axvline(0, color="#333333", ls="-", lw=1)
            ax.axvline(np.mean(asym_vals), color="#E04040", ls="--", lw=2,
                       label=f"Mean: {np.mean(asym_vals):.3f}")
            ax.legend(fontsize=9)

        ax.set_xlabel("ln(F4α) − ln(F3α)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Frontal Alpha Asymmetry — {cond}  (n={len(asym_vals)})",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out, "15_alpha_asymmetry")
    pd.DataFrame(csv_rows).to_csv(out / "15_alpha_asymmetry.csv", index=False)


def _ec_eo_alpha_suppression(spectra: SpectraResult, out: Path) -> None:
    """
    Paired scatter: EC alpha vs EO alpha per subject.
    Points above the diagonal indicate expected stronger EC alpha.
    """
    ec_keys = spectra.by_condition("EC")
    eo_keys = spectra.by_condition("EO")

    if not ec_keys or not eo_keys:
        return

    # Match subjects
    ec_by_sub = {spectra.subjects[k]: k for k in ec_keys}
    eo_by_sub = {spectra.subjects[k]: k for k in eo_keys}

    common_subs = sorted(set(ec_by_sub.keys()) & set(eo_by_sub.keys()))
    if not common_subs:
        return

    ec_alpha = []
    eo_alpha = []
    csv_rows = []

    for sub in common_subs:
        ec_psd = np.mean(spectra.psds[ec_by_sub[sub]], axis=0)
        eo_psd = np.mean(spectra.psds[eo_by_sub[sub]], axis=0)
        ec_bp = _band_power(spectra.freqs, ec_psd, *ALPHA_BAND)
        eo_bp = _band_power(spectra.freqs, eo_psd, *ALPHA_BAND)
        ec_alpha.append(np.log10(max(ec_bp, 1e-20)))
        eo_alpha.append(np.log10(max(eo_bp, 1e-20)))
        csv_rows.append({
            "sub": sub,
            "EC_alpha_log10": ec_alpha[-1],
            "EO_alpha_log10": eo_alpha[-1],
        })

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(eo_alpha, ec_alpha, alpha=0.6, color="#4A90D9", s=30, edgecolors="white")

    # Diagonal (identity line)
    lim_lo = min(min(eo_alpha), min(ec_alpha)) - 0.2
    lim_hi = max(max(eo_alpha), max(ec_alpha)) + 0.2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)

    # Correlation
    r = np.corrcoef(eo_alpha, ec_alpha)[0, 1]
    n_above = sum(1 for ec, eo in zip(ec_alpha, eo_alpha) if ec > eo)
    ax.set_xlabel("EO alpha power (log₁₀)", fontsize=11)
    ax.set_ylabel("EC alpha power (log₁₀)", fontsize=11)
    ax.set_title(
        f"EC vs EO Alpha Suppression  (n={len(common_subs)}, r={r:.3f})\n"
        f"{n_above}/{len(common_subs)} subjects with EC > EO",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    _save(fig, out, "16_ec_eo_alpha_suppression")
    pd.DataFrame(csv_rows).to_csv(out / "16_ec_eo_alpha_suppression.csv", index=False)


def generate_alpha_plots(
    spectra: SpectraResult,
    cohort_df: pd.DataFrame,
    out: Path,
) -> None:
    """Generate all alpha validation plots."""
    _alpha_topography(spectra, out)
    _iapf_distribution(spectra, out)
    _alpha_asymmetry_histogram(spectra, out)
    _ec_eo_alpha_suppression(spectra, out)
    print("  → Alpha plots: 13–16")
