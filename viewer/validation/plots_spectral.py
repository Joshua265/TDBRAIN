"""
Spectral validation plots — must-have 4.

Grand median PSD + IQR, condition-specific, region-specific.
"über alle Fenster und alle Probanden mal ein Median Spektrum abbilden
und den Interquantilbereich darstellen."
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spectral import SpectraResult, REGIONS


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _median_spectrum_plot(
    freqs: np.ndarray,
    psd_matrix: np.ndarray,
    title: str,
    ax: plt.Axes,
    fmax: float = 45.0,
    label: Optional[str] = None,
    color: str = "#2C5F8A",
    alpha_iqr: float = 0.25,
    alpha_5_95: float = 0.10,
) -> None:
    """
    Plot median PSD with IQR and 5th–95th percentile bands.

    psd_matrix: (n_subjects, n_freqs) — linear power.
    """
    keep = freqs <= fmax
    f = freqs[keep]
    mat = psd_matrix[:, keep]

    # Log10 transform for display
    mat_db = 10.0 * np.log10(np.maximum(mat, 1e-20))

    median = np.median(mat_db, axis=0)
    q25 = np.percentile(mat_db, 25, axis=0)
    q75 = np.percentile(mat_db, 75, axis=0)
    q05 = np.percentile(mat_db, 5, axis=0)
    q95 = np.percentile(mat_db, 95, axis=0)

    ax.fill_between(f, q05, q95, alpha=alpha_5_95, color=color, label="5th–95th pctl")
    ax.fill_between(f, q25, q75, alpha=alpha_iqr, color=color, label="IQR (25th–75th)")
    ax.plot(f, median, color=color, lw=2, label=label or "Median")

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Power (dB, 10·log₁₀ µV²/Hz)", fontsize=11)
    ax.grid(True, alpha=0.3)


def _grand_median_psd(spectra: SpectraResult, out: Path) -> None:
    """
    Grand median PSD across all windows, all subjects, all channels.
    Separate plot per condition.
    """
    conditions = sorted(set(spectra.conditions.values()))

    for cond in conditions:
        keys = spectra.by_condition(cond)
        if not keys:
            continue

        # Collect all channel-mean PSDs
        all_psds = []
        for k in keys:
            all_psds.append(np.mean(spectra.psds[k], axis=0))
        mat = np.vstack(all_psds)

        fig, ax = plt.subplots(figsize=(10, 6))
        _median_spectrum_plot(spectra.freqs, mat,
                              f"Grand Median PSD — {cond}  (n={len(keys)})", ax)
        ax.set_title(f"Grand Median PSD — {cond}  (n={len(keys)})",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        _save(fig, out, f"07_grand_median_psd_{cond}")

        # CSV: frequencies + median + quartiles
        keep = spectra.freqs <= 45.0
        f = spectra.freqs[keep]
        mat_db = 10.0 * np.log10(np.maximum(mat[:, keep], 1e-20))
        csv_df = pd.DataFrame({
            "freq_hz": f,
            "median_dB": np.median(mat_db, axis=0),
            "q25_dB": np.percentile(mat_db, 25, axis=0),
            "q75_dB": np.percentile(mat_db, 75, axis=0),
            "q05_dB": np.percentile(mat_db, 5, axis=0),
            "q95_dB": np.percentile(mat_db, 95, axis=0),
        })
        csv_df.to_csv(out / f"07_grand_median_psd_{cond}.csv", index=False)


def _condition_comparison(spectra: SpectraResult, out: Path) -> None:
    """
    EC vs EO spectra on the same axes.
    """
    conditions = sorted(set(spectra.conditions.values()))
    if len(conditions) < 2:
        return

    colors = {"EC": "#2C5F8A", "EO": "#D4651E"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in conditions:
        keys = spectra.by_condition(cond)
        if not keys:
            continue
        all_psds = [np.mean(spectra.psds[k], axis=0) for k in keys]
        mat = np.vstack(all_psds)
        c = colors.get(cond, "#666666")
        _median_spectrum_plot(spectra.freqs, mat, "", ax,
                              label=f"{cond} (n={len(keys)})", color=c)

    ax.set_title("Condition Comparison: EC vs EO", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out, "08_condition_comparison_psd")


def _regional_spectra(spectra: SpectraResult, out: Path) -> None:
    """
    Region-specific spectra: frontal, central, parietal, occipital.
    Separate figure per condition.
    """
    conditions = sorted(set(spectra.conditions.values()))
    region_colors = {
        "frontal": "#E04040",
        "central": "#4A90D9",
        "parietal": "#5CB85C",
        "occipital": "#9B59B6",
    }

    for cond in conditions:
        keys = spectra.by_condition(cond)
        if not keys:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for region_name, region_chs in REGIONS.items():
            region_psds = []
            for k in keys:
                rpsd = spectra.channel_psd_for_region(k, region_chs)
                if rpsd is not None:
                    region_psds.append(rpsd)

            if not region_psds:
                continue

            mat = np.vstack(region_psds)
            c = region_colors.get(region_name, "#888888")
            _median_spectrum_plot(
                spectra.freqs, mat, "", ax,
                label=f"{region_name} (n={len(region_psds)})", color=c,
            )

        ax.set_title(f"Regional Spectra — {cond}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        _save(fig, out, f"09_regional_spectra_{cond}")

        # CSV: per-region median
        keep = spectra.freqs <= 45.0
        f = spectra.freqs[keep]
        csv_data = {"freq_hz": f}
        for region_name, region_chs in REGIONS.items():
            region_psds = []
            for k in keys:
                rpsd = spectra.channel_psd_for_region(k, region_chs)
                if rpsd is not None:
                    region_psds.append(rpsd)
            if region_psds:
                mat = np.vstack(region_psds)
                mat_db = 10.0 * np.log10(np.maximum(mat[:, keep], 1e-20))
                csv_data[f"{region_name}_median_dB"] = np.median(mat_db, axis=0)
                csv_data[f"{region_name}_q25_dB"] = np.percentile(mat_db, 25, axis=0)
                csv_data[f"{region_name}_q75_dB"] = np.percentile(mat_db, 75, axis=0)
        pd.DataFrame(csv_data).to_csv(
            out / f"09_regional_spectra_{cond}.csv", index=False
        )


def generate_spectral_plots(
    spectra: SpectraResult, cohort_df: pd.DataFrame, out: Path
) -> None:
    """Generate all spectral validation plots."""
    _grand_median_psd(spectra, out)
    _condition_comparison(spectra, out)
    _regional_spectra(spectra, out)
    print("  → Spectral plots: 07–09")
