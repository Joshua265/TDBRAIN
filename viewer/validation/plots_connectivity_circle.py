"""Circle plots for pre/post rTMS channel power + coherence connectivity.

This approximates the manuscript-style Figure 2 visualization but uses EEG
channels (not source ROIs) as nodes.

- Power: paired t-test on per-channel bandpower (post - pre), FDR-corrected.
- Connectivity: coherence per band and condition; paired t-test on edges.
  Significant networks are detected with a simple sign-flip permutation NBS.

Outputs:
- One figure per band and condition: power bars + significant edges.
- A theta EC paired connectivity panel + edge-count table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

from .connectivity import CoherenceResult
from .outliers import _band_power
from .spectral import SpectraResult, REGIONS


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


def _bh_fdr(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR mask."""
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    n = flat.size
    order = np.argsort(flat)
    ranked = flat[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    ok = ranked <= thresh
    if not np.any(ok):
        return np.zeros_like(p, dtype=bool)
    k = np.max(np.where(ok)[0])
    cutoff = ranked[k]
    return p <= cutoff


def _paired_t(diff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Paired t-test on differences.

    Returns (t, p) with normal-approx p (sufficient for permutation NBS).
    """
    diff = np.asarray(diff, dtype=float)
    n = diff.shape[0]
    mean = diff.mean(axis=0)
    sd = diff.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, np.inf, sd)
    t = mean / (sd / np.sqrt(n))

    def norm_sf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 - erf(x / np.sqrt(2.0)))

    p = 2.0 * norm_sf(np.abs(t))
    return t, p


def _nbs_signflip(
    t_obs: np.ndarray,
    diffs: np.ndarray,
    t_threshold: float,
    n_perm: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Permutation NBS with sign-flip permutations (paired design).

    Returns a boolean adjacency mask of significant edges.

    Notes:
    - Uses component mass = sum(|t|) over edges in component.
    - Graph connectivity is defined on an undirected graph of suprathreshold edges.
    """

    rng = np.random.default_rng(seed)
    n_subj = diffs.shape[0]
    n_nodes = t_obs.shape[0]

    def suprathreshold_components(t_mat: np.ndarray) -> List[List[Tuple[int, int]]]:
        mask = np.abs(t_mat) >= t_threshold
        np.fill_diagonal(mask, False)
        visited = np.zeros_like(mask, dtype=bool)
        comps: List[List[Tuple[int, int]]] = []

        # Build adjacency list on the fly
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if not mask[i, j] or visited[i, j]:
                    continue
                # BFS on nodes, track edges
                queue = [i, j]
                nodes = set(queue)
                edges: List[Tuple[int, int]] = []
                while queue:
                    u = queue.pop()
                    for v in range(n_nodes):
                        a, b = (u, v) if u < v else (v, u)
                        if a == b:
                            continue
                        if not mask[a, b] or visited[a, b]:
                            continue
                        visited[a, b] = True
                        visited[b, a] = True
                        edges.append((a, b))
                        if v not in nodes:
                            nodes.add(v)
                            queue.append(v)
                if edges:
                    comps.append(edges)
        return comps

    comps = suprathreshold_components(t_obs)
    if not comps:
        return np.zeros_like(t_obs, dtype=bool)

    comp_masses = [float(np.sum(np.abs([t_obs[i, j] for i, j in edges])) ) for edges in comps]

    null_max = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_subj)
        diffs_p = diffs * signs[:, None, None]
        t_p, _ = _paired_t(diffs_p)
        comps_p = suprathreshold_components(t_p)
        if not comps_p:
            null_max.append(0.0)
            continue
        masses_p = [float(np.sum(np.abs([t_p[i, j] for i, j in edges])) ) for edges in comps_p]
        null_max.append(max(masses_p))

    null_max = np.asarray(null_max)
    sig = np.zeros_like(t_obs, dtype=bool)

    # component p-values
    for edges, mass in zip(comps, comp_masses):
        p = (np.sum(null_max >= mass) + 1.0) / (len(null_max) + 1.0)
        if p < 0.05:
            for i, j in edges:
                sig[i, j] = True
                sig[j, i] = True

    np.fill_diagonal(sig, False)
    return sig


def _channel_region(label: str) -> str:
    for region, chs in REGIONS.items():
        if label in chs:
            return region
    return "other"


def _layout_circle(labels: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """Return mapping label -> (angle_rad, x, y)."""
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # start at top, go clockwise
    ang = np.pi / 2 - ang
    return {lab: (a, float(np.cos(a)), float(np.sin(a))) for lab, a in zip(labels, ang)}


def _plot_circle_map(
    labels: List[str],
    power_t: np.ndarray,
    power_sig: np.ndarray,
    conn_t: np.ndarray,
    conn_sig: np.ndarray,
    title: str,
    out: Path,
    name: str,
) -> None:
    pos = _layout_circle(labels)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # Circle outline
    circ = plt.Circle((0, 0), 1.0, fill=False, lw=1.0, color="#999999")
    ax.add_patch(circ)

    # Power bars
    vmax = max(1e-9, float(np.nanmax(np.abs(power_t))))
    for i, lab in enumerate(labels):
        a, x, y = pos[lab]
        t = float(power_t[i])
        # bar length scaled
        L = 0.35 * (abs(t) / vmax)
        r0 = 1.02
        r1 = r0 + L
        x0, y0 = r0 * x, r0 * y
        x1, y1 = r1 * x, r1 * y
        color = "#D73027" if t > 0 else "#4575B4"
        alpha = 0.95 if bool(power_sig[i]) else 0.25
        ax.plot([x0, x1], [y0, y1], lw=4.0, color=color, alpha=alpha, solid_capstyle="round")

    # Node markers by region
    region_colors = {
        "frontal": "#1f77b4",
        "central": "#2ca02c",
        "parietal": "#9467bd",
        "occipital": "#ff7f0e",
        "other": "#7f7f7f",
    }
    for lab in labels:
        _, x, y = pos[lab]
        region = _channel_region(lab)
        ax.scatter([x], [y], s=20, c=region_colors.get(region, "#7f7f7f"), zorder=3)
        ax.text(1.12 * x, 1.12 * y, lab, fontsize=7, ha="center", va="center")

    # Connectivity edges
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if not conn_sig[i, j]:
                continue
            li, lj = labels[i], labels[j]
            _, xi, yi = pos[li]
            _, xj, yj = pos[lj]
            t = float(conn_t[i, j])
            color = "#D73027" if t > 0 else "#4575B4"
            ax.plot([xi, xj], [yi, yj], color=color, lw=1.2, alpha=0.8, zorder=1)

    ax.set_title(title, fontsize=13, fontweight="bold")
    _save(fig, out, name)


def _paired_channel_bandpower(
    spectra: SpectraResult,
    sub: str,
    ses: str,
    cond: str,
    flo: float,
    fhi: float,
) -> Optional[Tuple[List[str], np.ndarray]]:
    k = f"{sub}_{ses}_{cond}"
    psd = spectra.psds.get(k)
    labels = spectra.channel_labels.get(k)
    if psd is None or labels is None:
        return None
    vals = np.array([_band_power(spectra.freqs, psd[i], flo, fhi) for i in range(psd.shape[0])], dtype=float)
    return labels, vals


def generate_connectivity_circle_plots(
    spectra: SpectraResult,
    coh: CoherenceResult,
    cohort_df: pd.DataFrame,
    out: Path,
    pre_ses: str = "ses-1",
    post_ses: str = "ses-2",
    nbs_t_threshold: float = 2.5,
    nbs_perm: int = 500,
) -> None:
    """Generate circle maps per band and condition.

    Assumes `cohort_df` already contains only included recordings.
    Filters to MDD indication and subjects with both pre/post sessions.
    """

    df = cohort_df
    if "excluded" in df.columns:
        df = df[~df["excluded"].astype(bool)].copy()
    elif "qc_passed" in df.columns:
        df = df[df["qc_passed"].astype(bool)].copy()

    if "tsv_indication" not in df.columns:
        print("  [INFO] No tsv_indication column; skipping connectivity circle plots")
        return

    mdd = df[df["tsv_indication"].astype(str).str.upper().str.contains("MDD", na=False)]
    have_ses = mdd.groupby("sub")["ses"].apply(lambda x: set(x.astype(str))).to_dict()
    subs = sorted([s for s, ss in have_ses.items() if {pre_ses, post_ses}.issubset(ss)])
    if not subs:
        print("  [INFO] No eligible MDD subjects with pre/post sessions; skipping")
        return

    conditions = [c for c in ["EO", "EC"] if c in set(spectra.conditions.values())]
    if not conditions:
        return

    csv_rows: List[Dict[str, object]] = []

    for band_name, flo, fhi in BANDS:
        for cond in conditions:
            # collect paired power per channel
            pre_vals = []
            post_vals = []
            labels_ref: Optional[List[str]] = None

            # collect paired coherence matrices
            pre_conn = []
            post_conn = []

            for sub in subs:
                pre_bp = _paired_channel_bandpower(spectra, sub, pre_ses, cond, flo, fhi)
                post_bp = _paired_channel_bandpower(spectra, sub, post_ses, cond, flo, fhi)
                if pre_bp is None or post_bp is None:
                    continue

                labels, v_pre = pre_bp
                labels2, v_post = post_bp
                if labels != labels2:
                    continue

                k_pre = f"{sub}_{pre_ses}_{cond}"
                k_post = f"{sub}_{post_ses}_{cond}"
                m_pre = coh.band_mats.get(k_pre, {}).get(band_name)
                m_post = coh.band_mats.get(k_post, {}).get(band_name)
                if m_pre is None or m_post is None:
                    continue

                if labels_ref is None:
                    labels_ref = labels
                if labels_ref != labels:
                    continue

                pre_vals.append(v_pre)
                post_vals.append(v_post)
                pre_conn.append(m_pre)
                post_conn.append(m_post)

            if labels_ref is None or len(pre_vals) < 5:
                continue

            pre_arr = np.vstack(pre_vals)
            post_arr = np.vstack(post_vals)
            diff = post_arr - pre_arr
            power_t, power_p = _paired_t(diff)
            power_sig = _bh_fdr(power_p, alpha=0.05)

            pre_c = np.stack(pre_conn, axis=0)
            post_c = np.stack(post_conn, axis=0)
            diff_c = post_c - pre_c
            conn_t, _ = _paired_t(diff_c)
            conn_sig = _nbs_signflip(conn_t, diff_c, t_threshold=nbs_t_threshold, n_perm=nbs_perm, seed=0)

            title = f"{band_name} ({flo:g}-{fhi:g} Hz) — {cond}: Post - Pre"
            name = f"18_circle_{band_name}_{cond}".replace(" ", "_").replace("/", "-")
            _plot_circle_map(
                labels_ref,
                power_t,
                power_sig,
                conn_t,
                conn_sig,
                title=title,
                out=out,
                name=name,
            )

            # CSV summary
            for i, ch in enumerate(labels_ref):
                csv_rows.append(
                    {
                        "band": band_name,
                        "condition": cond,
                        "channel": ch,
                        "power_t": float(power_t[i]),
                        "power_p": float(power_p[i]),
                        "power_fdr_sig": bool(power_sig[i]),
                        "n_pairs": int(diff.shape[0]),
                    }
                )

            # Add theta EC paired connectivity panel + table
            if band_name == "Theta" and cond == "EC":
                _theta_ec_panel(labels_ref, subs, pre_ses, post_ses, coh, conn_sig, band_name, out)

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(out / "18_circle_power_stats.csv", index=False)
    print("  → Connectivity circle plots: 18")


def _theta_ec_panel(
    labels: List[str],
    subs_all: List[str],
    pre_ses: str,
    post_ses: str,
    coh: CoherenceResult,
    conn_sig: np.ndarray,
    band_name: str,
    out: Path,
) -> None:
    # extract edges
    edges = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if conn_sig[i, j]]
    if not edges:
        return

    pre_vals = []
    post_vals = []
    subs = []

    for sub in subs_all:
        k_pre = f"{sub}_{pre_ses}_EC"
        k_post = f"{sub}_{post_ses}_EC"
        m_pre = coh.band_mats.get(k_pre, {}).get(band_name)
        m_post = coh.band_mats.get(k_post, {}).get(band_name)
        if m_pre is None or m_post is None:
            continue
        epre = [m_pre[i, j] for i, j in edges]
        epost = [m_post[i, j] for i, j in edges]
        pre_vals.append(float(np.mean(epre)))
        post_vals.append(float(np.mean(epost)))
        subs.append(sub)

    if len(pre_vals) < 5:
        return

    pre_arr = np.array(pre_vals)
    post_arr = np.array(post_vals)

    fig = plt.figure(figsize=(10.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])

    # (b) paired plot
    ax = fig.add_subplot(gs[0, 0])
    for i in range(len(pre_arr)):
        ax.plot([0, 1], [pre_arr[i], post_arr[i]], color="#000000", lw=0.8, alpha=0.7)
        ax.scatter([0, 1], [pre_arr[i], post_arr[i]], s=12, color="#000000", alpha=0.7)

    parts = ax.violinplot([pre_arr, post_arr], positions=[0, 1], showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#cccccc")
        pc.set_alpha(0.35)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Post-rTMS"], fontsize=10)
    ax.set_ylabel("Mean coherence (significant network)")
    ax.set_title("Theta — Eyes Closed", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # (c) edge table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")

    region = {ch: _channel_region(ch) for ch in labels}

    def region_pair(a: str, b: str) -> str:
        ra, rb = region[a], region[b]
        if ra == rb:
            return f"Intra-{ra.capitalize()}"
        # order consistently
        x, y = sorted([ra, rb])
        return f"{x.capitalize()}–{y.capitalize()}"

    counts: Dict[str, int] = {}
    for i, j in edges:
        key = region_pair(labels[i], labels[j])
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    lines = ["Edge count", "", f"Total edges: {total}", "", "Locations of nodes\t n\t%"]
    for k in sorted(counts.keys()):
        n = counts[k]
        pct = 100.0 * n / max(total, 1)
        lines.append(f"{k}\t{n}\t{pct:4.0f}%")

    ax2.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#dddddd"),
    )

    fig.suptitle("Theta network summary (NBS p<0.05)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "19_theta_EC_network_summary")

    # CSV
    pd.DataFrame(
        {
            "sub": subs,
            "baseline_mean_coh": pre_arr,
            "post_mean_coh": post_arr,
        }
    ).to_csv(out / "19_theta_EC_network_summary.csv", index=False)
