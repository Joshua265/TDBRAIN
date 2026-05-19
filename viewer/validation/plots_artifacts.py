"""
Artifact validation plots — must-have 3.

- Artifact burden per subject (% rejected epochs, % rejected channels)
- Artifact type distribution
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _artifact_burden(df: pd.DataFrame, out: Path) -> None:
    """
    Bar chart: per-subject artifact burden.
    Shows % rejected epochs and artifact coverage % side by side.
    """
    conditions = sorted(df["condition"].unique())

    fig, axes = plt.subplots(len(conditions), 1,
                             figsize=(max(12, len(df) * 0.15), 5 * len(conditions)),
                             squeeze=False)

    csv_rows = []

    for ci, cond in enumerate(conditions):
        ax = axes[ci, 0]
        sub = df[df["condition"] == cond].sort_values("artifact_coverage_pct", ascending=False)

        if sub.empty:
            ax.set_title(f"Artifact Burden — {cond} (no data)")
            continue

        x = np.arange(len(sub))
        labels = sub["sub"].values
        pct_bad_epochs = 100.0 - sub["pct_good_epochs"].values
        pct_artifact = sub["artifact_coverage_pct"].values

        w = 0.35
        ax.bar(x - w / 2, pct_bad_epochs, w, label="% bad epochs", color="#E07040", alpha=0.85)
        ax.bar(x + w / 2, pct_artifact, w, label="% artifact coverage", color="#4A90D9", alpha=0.85)

        # Threshold lines
        ax.axhline(50, color="#B00020", ls="--", lw=1.5, label="50% reject threshold")
        ax.axhline(33.3, color="#FFA000", ls=":", lw=1.5, label="33% channel threshold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("Percent (%)", fontsize=11)
        ax.set_title(f"Artifact Burden per Subject — {cond}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        for _, r in sub.iterrows():
            csv_rows.append({
                "sub": r["sub"], "ses": r["ses"], "condition": cond,
                "pct_bad_epochs": 100.0 - r["pct_good_epochs"],
                "artifact_coverage_pct": r["artifact_coverage_pct"],
            })

    fig.tight_layout()
    _save(fig, out, "04_artifact_burden")
    pd.DataFrame(csv_rows).to_csv(out / "04_artifact_burden.csv", index=False)


def _artifact_type_distribution(df: pd.DataFrame, out: Path) -> None:
    """
    Stacked bar chart showing the count of each artifact detector type per subject.
    """
    det_cols = [c for c in df.columns if c.startswith("det_")]
    if not det_cols:
        print("  [INFO] No detector columns found, skipping artifact type distribution")
        return

    conditions = sorted(df["condition"].unique())

    fig, axes = plt.subplots(len(conditions), 1,
                             figsize=(max(12, len(df) * 0.15), 5 * len(conditions)),
                             squeeze=False)

    colors = plt.cm.Set2(np.linspace(0, 1, len(det_cols)))

    for ci, cond in enumerate(conditions):
        ax = axes[ci, 0]
        sub = df[df["condition"] == cond].copy()

        if sub.empty:
            ax.set_title(f"Artifact Types — {cond} (no data)")
            continue

        # Sort by total artifact count
        sub["_total"] = sub[det_cols].sum(axis=1)
        sub = sub.sort_values("_total", ascending=False)

        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))

        for j, col in enumerate(det_cols):
            vals = sub[col].fillna(0).values
            label = col.replace("det_", "")
            ax.bar(x, vals, bottom=bottom, label=label, color=colors[j], alpha=0.85,
                   edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(sub["sub"].values, rotation=90, fontsize=6)
        ax.set_ylabel("Artifact count", fontsize=11)
        ax.set_title(f"Artifact Type Distribution — {cond}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out, "05_artifact_type_distribution")
    df[["sub", "ses", "condition"] + det_cols].to_csv(
        out / "05_artifact_type_distribution.csv", index=False
    )


def _per_channel_burden_heatmap(
    per_channel_artifact: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
    out: Path,
) -> None:
    """
    Heatmap: subjects × channels showing per-channel artifact burden.
    """
    if not per_channel_artifact:
        return

    conditions = sorted(df["condition"].unique())

    for cond in conditions:
        sub_df = df[df["condition"] == cond]
        paths = sub_df["path"].values
        subs = sub_df["sub"].values

        # Collect all channel names
        all_chs = set()
        for p in paths:
            if str(p) in per_channel_artifact:
                all_chs.update(per_channel_artifact[str(p)].keys())
        if not all_chs:
            continue

        all_chs_sorted = sorted(all_chs)
        mat = np.full((len(paths), len(all_chs_sorted)), np.nan)

        for i, p in enumerate(paths):
            d = per_channel_artifact.get(str(p), {})
            for j, ch in enumerate(all_chs_sorted):
                if ch in d:
                    mat[i, j] = d[ch] * 100.0  # to percent

        fig, ax = plt.subplots(figsize=(max(10, len(all_chs_sorted) * 0.4),
                                        max(6, len(paths) * 0.15)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100,
                       interpolation="nearest")
        ax.set_xticks(range(len(all_chs_sorted)))
        ax.set_xticklabels(all_chs_sorted, rotation=90, fontsize=7)
        ax.set_yticks(range(len(subs)))
        ax.set_yticklabels(subs, fontsize=6)
        ax.set_xlabel("Channel", fontsize=11)
        ax.set_ylabel("Subject", fontsize=11)
        ax.set_title(
            f"Per-Channel Artifact Burden (%) — {cond}",
            fontsize=13, fontweight="bold",
        )
        plt.colorbar(im, ax=ax, label="% bad epochs", shrink=0.8)

        fig.tight_layout()
        _save(fig, out, f"06_channel_burden_heatmap_{cond}")

        # CSV
        csv_df = pd.DataFrame(mat, columns=all_chs_sorted)
        csv_df.insert(0, "sub", subs)
        csv_df.to_csv(out / f"06_channel_burden_heatmap_{cond}.csv", index=False)


def generate_artifact_plots(
    cohort_df: pd.DataFrame,
    per_channel_artifact: Dict[str, Dict[str, float]],
    out: Path,
) -> None:
    """Generate all artifact validation plots and CSVs."""
    _artifact_burden(cohort_df, out)
    _artifact_type_distribution(cohort_df, out)
    _per_channel_burden_heatmap(per_channel_artifact, cohort_df, out)
    print("  → Artifact plots: 04–06")
