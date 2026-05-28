"""
Artifact validation plots — must-have 3.

- Artifact burden per subject (% rejected epochs, % rejected channels)
- Artifact type distribution
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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


def _per_channel_burden_boxplot(
    per_channel_artifact: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
    out: Path,
) -> None:
    """Boxplots: per-channel distribution of bad-epoch burden across subjects."""
    if not per_channel_artifact:
        return

    conditions = sorted(df["condition"].unique())

    for cond in conditions:
        sub_df = df[df["condition"] == cond]
        paths = sub_df["path"].values
        subs = sub_df["sub"].values

        # Collect burdens in long-form rows
        rows = []
        channel_to_vals: Dict[str, List[float]] = {}

        for sub, p in zip(subs, paths):
            d = per_channel_artifact.get(str(p), {})
            for ch, burden in d.items():
                pct_bad = float(burden) * 100.0
                rows.append({"sub": sub, "condition": cond, "channel": ch, "pct_bad_epochs": pct_bad})
                channel_to_vals.setdefault(ch, []).append(pct_bad)

        if not channel_to_vals:
            continue

        channels = sorted(channel_to_vals.keys())
        data = [channel_to_vals[ch] for ch in channels]

        fig, ax = plt.subplots(
            figsize=(max(12, len(channels) * 0.35), 6)
        )
        bp = ax.boxplot(
            data,
            labels=channels,
            vert=True,
            patch_artist=True,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 2,
                "markerfacecolor": "#E04040",
                "markeredgecolor": "none",
                "alpha": 0.6,
            },
        )

        for box in bp["boxes"]:
            box.set_facecolor("#4A90D9")
            box.set_alpha(0.5)
            box.set_edgecolor("#2C5F8A")

        # Overlay explicit Tukey outliers (so they are visible even with styling)
        rng = np.random.default_rng(0)
        for xi, vals in enumerate(data, start=1):
            arr = np.asarray(vals, dtype=float)
            if arr.size < 4:
                continue
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outs = arr[(arr < lo) | (arr > hi)]
            if outs.size == 0:
                continue
            xj = xi + rng.uniform(-0.08, 0.08, size=outs.size)
            ax.scatter(
                xj,
                outs,
                s=10,
                c="#E04040",
                alpha=0.75,
                linewidths=0,
                zorder=3,
                label="outliers" if xi == 1 else None,
            )

        ax.axhline(33.3, color="#FFA000", ls=":", lw=1.5, label="33% channel threshold")
        ax.set_ylabel("% bad epochs", fontsize=11)
        ax.set_xlabel("Channel", fontsize=11)
        ax.set_title(
            f"Per-Channel Artifact Burden (Boxplots) — {cond}",
            fontsize=13,
            fontweight="bold",
        )
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        _save(fig, out, f"06_channel_burden_boxplot_{cond}")

        pd.DataFrame(rows).to_csv(out / f"06_channel_burden_boxplot_{cond}.csv", index=False)


def _per_channel_burden_heatmap_thresholded(
    per_channel_artifact: Dict[str, Dict[str, float]],
    df: pd.DataFrame,
    out: Path,
) -> None:
    """Heatmap: subjects×channels for subjects exceeding median burden.

    - Subject is included if at least one channel exceeds the condition-specific
      median (% bad epochs) across all subject-channel entries.
    - Channels are sorted by mean % bad epochs (descending).
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
            d = per_channel_artifact.get(str(p))
            if d:
                all_chs.update(d.keys())
        if not all_chs:
            continue

        all_chs_sorted = sorted(all_chs)
        mat = np.full((len(paths), len(all_chs_sorted)), np.nan)

        for i, p in enumerate(paths):
            d = per_channel_artifact.get(str(p), {})
            for j, ch in enumerate(all_chs_sorted):
                if ch in d:
                    mat[i, j] = float(d[ch]) * 100.0

        # Manual threshold; cohort median can be too low.
        median_thresh = 33.3

        row_keep = []
        for i in range(mat.shape[0]):
            row = mat[i]
            if np.all(~np.isfinite(row)):
                continue
            if np.nanmax(row) > median_thresh:
                row_keep.append(i)

        if not row_keep:
            continue

        mat_sel = mat[row_keep, :]
        subs_sel = subs[row_keep]

        ch_mean = np.nanmean(mat_sel, axis=0)
        order = np.argsort(np.where(np.isfinite(ch_mean), ch_mean, -np.inf))[::-1]
        mat_sel = mat_sel[:, order]
        ch_sorted = [all_chs_sorted[i] for i in order]

        fig, ax = plt.subplots(
            figsize=(max(10, len(ch_sorted) * 0.4), max(6, len(subs_sel) * 0.15))
        )
        im = ax.imshow(mat_sel, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks(range(len(ch_sorted)))
        ax.set_xticklabels(ch_sorted, rotation=90, fontsize=7)
        ax.set_yticks(range(len(subs_sel)))
        ax.set_yticklabels(subs_sel, fontsize=6)
        ax.set_xlabel("Channel (sorted by mean burden)", fontsize=11)
        ax.set_ylabel("Subject", fontsize=11)
        ax.set_title(
            f"Per-Channel Artifact Burden (%) — {cond}\n"
            f"Showing subjects with any channel > {median_thresh:.1f}%",
            fontsize=13,
            fontweight="bold",
        )
        plt.colorbar(im, ax=ax, label="% bad epochs", shrink=0.8)

        fig.tight_layout()
        _save(fig, out, f"06_channel_burden_heatmap_{cond}")

        csv_df = pd.DataFrame(mat_sel, columns=ch_sorted)
        csv_df.insert(0, "sub", subs_sel)
        csv_df.insert(1, "threshold_pct", median_thresh)
        csv_df.to_csv(out / f"06_channel_burden_heatmap_{cond}.csv", index=False)


def generate_artifact_plots(
    cohort_df: pd.DataFrame,
    per_channel_artifact: Dict[str, Dict[str, float]],
    out: Path,
) -> None:
    """Generate all artifact validation plots and CSVs."""
    _artifact_burden(cohort_df, out)
    _artifact_type_distribution(cohort_df, out)
    _per_channel_burden_boxplot(per_channel_artifact, cohort_df, out)
    _per_channel_burden_heatmap_thresholded(per_channel_artifact, cohort_df, out)
    print("  → Artifact plots: 04–06")
