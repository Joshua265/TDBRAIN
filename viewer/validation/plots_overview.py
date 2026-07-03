"""
Dataset overview plots — must-haves 1–2.

1. Subject inclusion / exclusion flowchart
2. Good epochs per subject histogram
3. Good channels per subject histogram
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _flowchart(df: pd.DataFrame, out: Path) -> None:
    """
    Text-based inclusion flowchart rendered as a matplotlib figure.
    """
    total = len(df)
    has_mask = df["has_artifact_mask"].sum()
    no_mask = total - has_mask

    # Rejection rules (matches preprocessing QC in `validation/scanner.py`).
    if "qc_passed" in df.columns:
        usable = df[df["qc_passed"].astype(bool)]
    else:
        usable = df[(df["pct_good_epochs"] >= 50) & (df["n_good_channels"] >= 20)]

    rejected_quality_bad = df[df.get("reject_tdbrain_quality_bad", False).astype(bool)]
    rejected_epochs = df[df.get("reject_bad_epochs", False).astype(bool)]
    rejected_channels = df[df.get("reject_too_few_channels", False).astype(bool)]
    rejected_no_epochs = df[df.get("reject_no_epochs", False).astype(bool)]

    # Per condition
    conditions = sorted(df["condition"].unique())
    cond_lines = []
    for c in conditions:
        sub = df[df["condition"] == c]
        u = usable[usable["condition"] == c]
        cond_lines.append(f"  {c}: {len(sub)} total → {len(u)} usable")

    lines = [
        f"Total recordings scanned: {total}",
        "",
        f"  With artifact mask:    {has_mask}",
        f"  Without artifact mask: {no_mask}",
        "",
        f"Rejected (pipeline flag bad):   {len(rejected_quality_bad)}",
        f"Rejected (<50% good epochs):    {len(rejected_epochs)}",
        f"Rejected (<20 good channels):   {len(rejected_channels)}",
        f"Rejected (no full epochs):      {len(rejected_no_epochs)}",
        "",
        f"Usable for analysis: {len(usable)}",
        "",
        "Per condition:",
    ] + cond_lines

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )
    ax.set_title("Subject Inclusion / Exclusion", fontsize=14, fontweight="bold", pad=20)
    _save(fig, out, "01_inclusion_flowchart")

    # Also save CSV
    summary = pd.DataFrame({
        "metric": [
            "total_recordings",
            "with_artifact_mask",
            "without_artifact_mask",
            "rejected_quality_bad",
            "rejected_epochs_lt50pct_good",
            "rejected_channels_lt20",
            "rejected_no_epochs",
            "usable",
        ],
        "count": [
            total,
            int(has_mask),
            int(no_mask),
            len(rejected_quality_bad),
            len(rejected_epochs),
            len(rejected_channels),
            len(rejected_no_epochs),
            len(usable),
        ],
    })
    summary.to_csv(out / "01_inclusion_flowchart.csv", index=False)


def _good_epochs_hist(df: pd.DataFrame, out: Path) -> None:
    """Histogram of good epochs per subject, split by condition."""
    conditions = sorted(df["condition"].unique())

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5),
                             squeeze=False)

    for i, cond in enumerate(conditions):
        ax = axes[0, i]
        sub = df[df["condition"] == cond]
        vals = sub["n_good_epochs"].values

        ax.hist(vals, bins=30, color="#4A90D9", edgecolor="white", alpha=0.85)

        # 50% threshold line
        if "n_total_epochs" in sub.columns:
            median_total = sub["n_total_epochs"].median()
            threshold = median_total * 0.5
            ax.axvline(threshold, color="#E04040", ls="--", lw=2,
                       label=f"50% of median total ({threshold:.0f})")
            ax.legend(fontsize=9)

        ax.set_xlabel("Number of good epochs", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Good Epochs — {cond}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out, "02_good_epochs_histogram")
    df[["sub", "ses", "condition", "n_total_epochs", "n_good_epochs", "pct_good_epochs"]].to_csv(
        out / "02_good_epochs_histogram.csv", index=False
    )


def _good_channels_hist(df: pd.DataFrame, out: Path) -> None:
    """Histogram of good channels per subject with threshold at 20."""
    conditions = sorted(df["condition"].unique())

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5),
                             squeeze=False)

    for i, cond in enumerate(conditions):
        ax = axes[0, i]
        sub = df[df["condition"] == cond]
        vals = sub["n_good_channels"].values

        ax.hist(vals, bins=range(0, int(vals.max()) + 2), color="#5CB85C",
                edgecolor="white", alpha=0.85)
        ax.axvline(20, color="#E04040", ls="--", lw=2, label="Threshold = 20")
        ax.legend(fontsize=9)

        ax.set_xlabel("Number of good channels", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Good Channels — {cond}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out, "03_good_channels_histogram")
    df[["sub", "ses", "condition", "n_eeg_channels", "n_good_channels"]].to_csv(
        out / "03_good_channels_histogram.csv", index=False
    )


def generate_overview_plots(
    cohort_df_full: pd.DataFrame,
    cohort_df_valid: pd.DataFrame,
    out: Path,
) -> None:
    """Generate all overview plots and CSVs.
    
    Uses cohort_df_full for the flowchart (which reports exclusion counts),
    and cohort_df_valid (=non-excluded) for histograms.
    """
    _flowchart(cohort_df_full, out)
    _good_epochs_hist(cohort_df_valid, out)
    _good_channels_hist(cohort_df_valid, out)
    print("  → Overview plots: 01–03")
