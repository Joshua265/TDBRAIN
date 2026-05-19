"""
CLI entry point for TDBRAIN validation pipeline.

Usage:
    python -m validation --root <dataset_root> --output <output_dir> [--participants-tsv <path>]

Mirrors the viewer's argument structure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python -m validation` from the viewer/ directory
# by making eeg_viewer importable.
_viewer_root = Path(__file__).resolve().parent.parent
if str(_viewer_root) not in sys.path:
    sys.path.insert(0, str(_viewer_root))

from eeg_viewer.io import scan_recordings

from .scanner import scan_cohort
from .plots_overview import generate_overview_plots
from .plots_artifacts import generate_artifact_plots
from .spectral import compute_cohort_spectra
from .plots_spectral import generate_spectral_plots
from .outliers import compute_outlier_scores
from .plots_outliers import generate_outlier_plots
from .plots_alpha import generate_alpha_plots


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TDBRAIN validation: cohort-level QC, spectral & alpha analysis"
    )
    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help="Dataset root to recursively scan for .npy recordings",
    )
    ap.add_argument(
        "--participants-tsv",
        type=str,
        default=None,
        help="participants.tsv with column participants_ID",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for PNGs and CSVs",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tsv = (
        Path(args.participants_tsv).expanduser().resolve()
        if args.participants_tsv
        else None
    )

    print(f"[validation] Scanning recordings under: {root}")
    recordings = scan_recordings(root)
    if not recordings:
        print("[validation] No recordings found. Check --root path.")
        sys.exit(1)

    n_files = sum(len(v) for v in recordings.values())
    print(f"[validation] Found {len(recordings)} subject/session entries, {n_files} files")

    # ── Phase 1: Scan cohort metadata ────────────────────────────────────
    print("\n[validation] Phase 1: Scanning cohort metadata …")
    cohort_df, per_channel_artifact = scan_cohort(recordings, tsv_path=tsv)
    cohort_df.to_csv(out / "cohort_metadata.csv", index=False)
    print(f"  → {len(cohort_df)} recordings scanned")

    # ── Phase 2: Overview + artifact plots (must-haves 1–3) ──────────────
    print("\n[validation] Phase 2: Overview & artifact plots …")
    generate_overview_plots(cohort_df, out)
    generate_artifact_plots(cohort_df, per_channel_artifact, out)

    # ── Phase 3: Spectral analysis (must-haves 4–5) ─────────────────────
    print("\n[validation] Phase 3: Computing cohort spectra …")
    spectra = compute_cohort_spectra(recordings, cohort_df)
    generate_spectral_plots(spectra, cohort_df, out)

    print("\n[validation] Phase 4: Spectral outlier analysis …")
    outlier_df = compute_outlier_scores(spectra, cohort_df)
    outlier_df.to_csv(out / "spectral_outliers.csv", index=False)
    generate_outlier_plots(spectra, cohort_df, outlier_df, out)

    # ── Phase 4: Alpha validation (must-have 6) ──────────────────────────
    print("\n[validation] Phase 5: Alpha validation …")
    generate_alpha_plots(spectra, cohort_df, out)

    print(f"\n[validation] Done. All outputs in: {out}")


if __name__ == "__main__":
    main()
