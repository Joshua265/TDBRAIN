"""Pre vs post rTMS bandpower plots.

Creates paired (within-subject) pre/post bandpower plots for participants
with an MDD indication and both `ses-1` and `ses-2` recordings.

This is meant as a lightweight analogue of typical "Baseline vs Post-TMS"
figures, using bandpower derived from Welch PSDs in `SpectraResult`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .outliers import _band_power
from .spectral import SpectraResult


BANDS: List[Tuple[str, float, float]] = [
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta-low", 13.0, 16.0),
    ("Beta-mid", 16.0, 20.0),
    ("Beta-high", 20.0, 30.0),
    ("Gamma-low", 30.0, 40.0),
    ("Gamma-high", 60.0, 80.0),
]


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _is_mdd_indication(val: object) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip().upper()
    return "MDD" in s


def _eligible_subjects(cohort_df: pd.DataFrame) -> List[str]:
    """Return subjects with MDD indication and both ses-1 & ses-2."""
    if "tsv_indication" not in cohort_df.columns:
        return []

    df = cohort_df[cohort_df["tsv_indication"].apply(_is_mdd_indication)].copy()
    if df.empty:
        return []

    have_ses = df.groupby("sub")["ses"].apply(lambda x: set(x.astype(str))).to_dict()
    eligible = [sub for sub, ses_set in have_ses.items() if {"ses-1", "ses-2"}.issubset(ses_set)]
    return sorted(eligible)


def _bandpower_from_key(spectra: SpectraResult, rec_key: str, flo: float, fhi: float) -> Optional[float]:
    psd = spectra.psds.get(rec_key)
    if psd is None or psd.size == 0:
        return None
    mean_psd = np.mean(psd, axis=0)
    return _band_power(spectra.freqs, mean_psd, flo, fhi)


def generate_tms_prepost_bandpower_plots(
    spectra: SpectraResult,
    cohort_df: pd.DataFrame,
    out: Path,
    pre_ses: str = "ses-1",
    post_ses: str = "ses-2",
) -> None:
    """Generate a multi-panel pre/post bandpower plot and CSV.

    Filters to:
    - `tsv_indication` contains "MDD"
    - has both `pre_ses` and `post_ses` sessions

    Creates one row per band, and two columns for conditions (EO/EC).
    Each panel shows paired subject lines (log10 bandpower).
    """

    df = cohort_df
    if "excluded" in df.columns:
        df = df[~df["excluded"].astype(bool)].copy()
    elif "qc_passed" in df.columns:
        df = df[df["qc_passed"].astype(bool)].copy()

    subjects = _eligible_subjects(df)
    if not subjects:
        print("  [INFO] No eligible MDD subjects with pre+post sessions found")
        return

    conditions = [c for c in ["EO", "EC"] if c in set(spectra.conditions.values())]
    if not conditions:
        print("  [INFO] No EO/EC spectra available for pre/post plot")
        return

    rows: List[Dict[str, object]] = []

    # Collect paired values
    paired: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = {}
    # keyed by (band_name, condition) -> list of (sub, pre, post)

    for band_name, flo, fhi in BANDS:
        for cond in conditions:
            pairs: List[Tuple[str, float, float]] = []
            for sub in subjects:
                k_pre = f"{sub}_{pre_ses}_{cond}"
                k_post = f"{sub}_{post_ses}_{cond}"
                pre = _bandpower_from_key(spectra, k_pre, flo, fhi)
                post = _bandpower_from_key(spectra, k_post, flo, fhi)
                if pre is None or post is None:
                    continue
                pairs.append((sub, float(pre), float(post)))
                rows.append(
                    {
                        "sub": sub,
                        "condition": cond,
                        "band": band_name,
                        "flo_hz": flo,
                        "fhi_hz": fhi,
                        "pre_power": float(pre),
                        "post_power": float(post),
                        "pre_log10": float(np.log10(max(pre, 1e-20))),
                        "post_log10": float(np.log10(max(post, 1e-20))),
                    }
                )
            paired[(band_name, cond)] = pairs

    if not rows:
        print("  [INFO] No complete pre/post pairs found in spectra")
        return

    # Plot
    n_rows = len(BANDS)
    n_cols = len(conditions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 1.9 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    for ri, (band_name, flo, fhi) in enumerate(BANDS):
        for ci, cond in enumerate(conditions):
            ax = axes[ri, ci]
            pairs = paired.get((band_name, cond), [])
            if not pairs:
                ax.set_axis_off()
                continue

            pre_vals = np.array([p for _, p, _ in pairs], dtype=float)
            post_vals = np.array([q for _, _, q in pairs], dtype=float)
            pre_log = np.log10(np.maximum(pre_vals, 1e-20))
            post_log = np.log10(np.maximum(post_vals, 1e-20))

            # Paired lines
            for i in range(len(pairs)):
                ax.plot([0, 1], [pre_log[i], post_log[i]], color="#000000", lw=0.8, alpha=0.65)

            # Mean marker
            ax.scatter([0, 1], [pre_log.mean(), post_log.mean()], color="#E04040", s=18, zorder=3)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pre", "Post"], fontsize=9)
            if ri == 0:
                ax.set_title(f"{cond}", fontsize=12, fontweight="bold")

            ax.grid(axis="y", alpha=0.25)
            ax.set_ylabel(f"{band_name}\nlog10(power)", fontsize=9)

            n = len(pairs)
            ax.text(
                0.02,
                0.95,
                f"n={n}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#444444",
            )

    fig.suptitle(
        "Baseline (ses-1) vs Post-rTMS (ses-2) — Bandpower (MDD only)",
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )
    fig.tight_layout()
    _save(fig, out, "17_pre_post_rTMS_bandpower_MDD")

    pd.DataFrame(rows).to_csv(out / "17_pre_post_rTMS_bandpower_MDD.csv", index=False)
    print("  → Pre/Post rTMS bandpower plot: 17")
