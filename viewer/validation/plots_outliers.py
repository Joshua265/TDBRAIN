"""
Spectral outlier plots — must-have 5 (visualization).

- Spectral distance per subject
- Outlier overlay (top N extreme spectra against cohort median)
- Bandpower boxplots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spectral import SpectraResult
from .outliers import BANDS, _band_power


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _spectral_distance_plot(
    outlier_df: pd.DataFrame, out: Path
) -> None:
    """One point per subject: spectral distance from cohort median."""
    conditions = sorted(outlier_df["condition"].unique())

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(7 * len(conditions), 5), squeeze=False)

    for i, cond in enumerate(conditions):
        ax = axes[0, i]
        sub = outlier_df[outlier_df["condition"] == cond].sort_values(
            "spectral_distance", ascending=True
        )
        if sub.empty:
            continue

        x = np.arange(len(sub))
        ax.barh(x, sub["spectral_distance"].values, color="#4A90D9", alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(sub["sub"].values, fontsize=6)
        ax.set_xlabel("Spectral distance (MAD-normalized)", fontsize=11)
        ax.set_title(f"Spectral Distance — {cond}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Highlight top 10
        top10 = sub.tail(10)
        for j, (_, r) in enumerate(top10.iterrows()):
            idx = np.where(sub["sub"].values == r["sub"])[0]
            if len(idx):
                ax.barh(idx[0], r["spectral_distance"], color="#E04040", alpha=0.8)

    fig.tight_layout()
    _save(fig, out, "10_spectral_distance")


def _outlier_overlay(
    spectra: SpectraResult,
    outlier_df: pd.DataFrame,
    out: Path,
    n_top: int = 10,
) -> None:
    """
    Cohort median + IQR with top N most extreme subject spectra overlaid.
    """
    conditions = sorted(set(spectra.conditions.values()))
    fmax = 100.0

    for cond in conditions:
        keys = spectra.by_condition(cond)
        if len(keys) < 3:
            continue

        # Build mean-PSD matrix
        key_to_idx = {k: i for i, k in enumerate(keys)}
        all_psds = [np.mean(spectra.psds[k], axis=0) for k in keys]
        mat = np.vstack(all_psds)
        mat_db = 10.0 * np.log10(np.maximum(mat, 1e-20))
        freqs = spectra.freqs
        keep = freqs <= fmax
        f = freqs[keep]

        cohort_median = np.median(mat_db[:, keep], axis=0)
        q25 = np.percentile(mat_db[:, keep], 25, axis=0)
        q75 = np.percentile(mat_db[:, keep], 75, axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(f, q25, q75, alpha=0.2, color="#888888", label="IQR")
        ax.plot(f, cohort_median, color="#333333", lw=2, label="Cohort median")

        # Top N outliers
        cond_outliers = outlier_df[outlier_df["condition"] == cond].head(n_top)
        cmap = plt.cm.hot(np.linspace(0.2, 0.8, n_top))

        for j, (_, row) in enumerate(cond_outliers.iterrows()):
            k = row["rec_key"]
            if k in key_to_idx:
                idx = key_to_idx[k]
                ax.plot(f, mat_db[idx, keep], color=cmap[j], lw=1, alpha=0.7,
                        label=f"{row['sub']} ({row['main_reason'][:30]})")

        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("Power (dB)", fontsize=11)
        ax.set_title(f"Outlier Overlay — {cond} (top {n_top})",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        _save(fig, out, f"11_outlier_overlay_{cond}")


def _bandpower_boxplots(
    spectra: SpectraResult,
    outlier_df: pd.DataFrame,
    out: Path,
) -> None:
    """
    Boxplots of bandpower per subject, with outlier subjects labeled.
    """
    conditions = sorted(set(spectra.conditions.values()))

    for cond in conditions:
        keys = spectra.by_condition(cond)
        if not keys:
            continue

        freqs = spectra.freqs
        band_names = list(BANDS.keys())

        # Compute bandpower per subject
        bp_data = {b: [] for b in band_names}
        subs = []
        for k in keys:
            mean_psd = np.mean(spectra.psds[k], axis=0)
            for b, (flo, fhi) in BANDS.items():
                bp_data[b].append(_band_power(freqs, mean_psd, flo, fhi))
            subs.append(spectra.subjects.get(k, "?"))

        fig, axes = plt.subplots(1, len(band_names),
                                 figsize=(3 * len(band_names), 5), squeeze=False)

        for j, b in enumerate(band_names):
            ax = axes[0, j]
            vals = np.array(bp_data[b])
            log_vals = np.log10(np.maximum(vals, 1e-20))

            bp = ax.boxplot(log_vals, vert=True, patch_artist=True)
            bp["boxes"][0].set_facecolor("#4A90D9")
            bp["boxes"][0].set_alpha(0.5)

            # Mark top 5 outliers
            top5 = outlier_df[outlier_df["condition"] == cond].head(5)
            for _, row in top5.iterrows():
                k = row["rec_key"]
                if k in keys:
                    idx = keys.index(k)
                    ax.plot(1, log_vals[idx], "ro", markersize=5, alpha=0.7)
                    ax.annotate(row["sub"], (1, log_vals[idx]), fontsize=5,
                               xytext=(5, 0), textcoords="offset points")

            ax.set_title(b.capitalize(), fontsize=11, fontweight="bold")
            ax.set_ylabel("log₁₀ power" if j == 0 else "")
            ax.set_xticklabels([])
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Bandpower Distribution — {cond}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        _save(fig, out, f"12_bandpower_boxplots_{cond}")

        # CSV
        csv_rows = []
        for i, k in enumerate(keys):
            row = {"sub": subs[i], "condition": cond, "rec_key": k}
            for b in band_names:
                row[f"{b}_power"] = bp_data[b][i]
                row[f"{b}_log10"] = np.log10(max(bp_data[b][i], 1e-20))
            csv_rows.append(row)
        pd.DataFrame(csv_rows).to_csv(
            out / f"12_bandpower_boxplots_{cond}.csv", index=False
        )


def generate_outlier_plots(
    spectra: SpectraResult,
    cohort_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    out: Path,
) -> None:
    """Generate all outlier visualization plots.

    Uses only included recordings if `excluded`/`qc_passed` is present.
    """

    if outlier_df.empty:
        print("  [INFO] No outlier data, skipping outlier plots")
        return

    df = cohort_df
    if "excluded" in df.columns:
        df = df[~df["excluded"].astype(bool)].copy()
    elif "qc_passed" in df.columns:
        df = df[df["qc_passed"].astype(bool)].copy()

    out_df = outlier_df
    if not df.empty and "sub" in out_df.columns and "condition" in out_df.columns:
        allowed = set(zip(df["sub"].astype(str), df["condition"].astype(str)))
        out_df = out_df[
            out_df.apply(
                lambda r: (str(r.get("sub")), str(r.get("condition"))) in allowed,
                axis=1,
            )
        ].copy()

    if out_df.empty:
        print("  [INFO] No included outlier data, skipping outlier plots")
        return

    _spectral_distance_plot(out_df, out)
    _outlier_overlay(spectra, out_df, out)
    _bandpower_boxplots(spectra, out_df, out)
    print("  → Outlier plots: 10–12")
