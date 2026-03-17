from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from ..qt_compat import QtCore, QtWidgets
from ..artifacts import ArtifactModel, mask_to_segments
from ..io import RecordingKey, load_eeg_dict
from .base import ViewBase


# ---------------------------------------------------------------------------
# Tmp-dir cache helpers
# ---------------------------------------------------------------------------

_TMP_DIR = Path(tempfile.gettempdir()) / "passage_hist_cache"


def _cache_path(path: Path) -> Path:
    """Return a deterministic .npy cache path for a recording file."""
    key = hashlib.md5(str(path.resolve()).encode()).hexdigest()
    return _TMP_DIR / f"{key}.npy"


def _load_cached(path: Path) -> Optional[np.ndarray]:
    cp = _cache_path(path)
    if cp.exists():
        try:
            return np.load(str(cp))
        except Exception:
            pass
    return None


def _save_cached(path: Path, lengths: np.ndarray) -> None:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        np.save(str(_cache_path(path)), lengths)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core computation (per file)
# ---------------------------------------------------------------------------

def _passages_from_path(path: Path) -> Optional[np.ndarray]:
    """
    Load a recording, build an ArtifactModel (identical approach to pcorr.py),
    and return artifact-free passage lengths in seconds.

    Returns None on load error. Returns a zero-element array if fully artifacted.
    Result is cached to *_TMP_DIR* so subsequent calls are instant.
    """
    cached = _load_cached(path)
    if cached is not None:
        return cached

    try:
        eeg = load_eeg_dict(path)
    except Exception:
        return None

    fs = int(eeg.get("Fs", 1))
    data = np.squeeze(np.asarray(eeg.get("data")))
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        return None

    n_samp = data.shape[1]

    labels = eeg.get("labels")
    if isinstance(labels, np.ndarray):
        labels = [str(x) for x in labels.tolist()]
    elif isinstance(labels, list):
        labels = [str(x) for x in labels]
    else:
        labels = [f"ch{i}" for i in range(data.shape[0])]

    # Build ArtifactModel — same approach as pcorr.py
    artifact_model = ArtifactModel(eeg, data, labels, fs)

    if artifact_model.artifact_mask is not None:
        clean = ~artifact_model.artifact_mask
        segs = mask_to_segments(clean)
    else:
        segs = np.array([[0, n_samp]], dtype=np.int64)

    if segs.size == 0:
        result = np.array([], dtype=np.float64)
    else:
        result = (segs[:, 1] - segs[:, 0]).astype(np.float64) / float(fs)

    _save_cached(path, result)
    return result


# ---------------------------------------------------------------------------
# Background scan worker
# ---------------------------------------------------------------------------

class _ScanWorker(QtCore.QThread):
    """
    Background thread: scans EO and EC paths for ses-1 subjects.

    Accepts two dicts {sub: Path} — one for Eyes-Open, one for Eyes-Closed.
    Emits (combined_lengths, per_eo, per_ec, n_total_subjects) where
    per_eo/per_ec are lists of per-subject passage-length arrays (only
    subjects for which that condition was found). n_total_subjects is the
    union of subjects seen across both conditions.
    """

    progress = QtCore.Signal(int, int)   # (done, total)
    # finished: (eo_map, ec_map, n_total_subjects)
    finished = QtCore.Signal(object, object, int)

    def __init__(
        self,
        eo_paths: Dict[str, Path],
        ec_paths: Dict[str, Path],
        parent=None,
    ):
        super().__init__(parent)
        self._eo_paths = eo_paths  # {sub: path}
        self._ec_paths = ec_paths  # {sub: path}
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        all_subjects = sorted(set(self._eo_paths) | set(self._ec_paths))
        n_total = len(all_subjects)

        # Process all paths; each subject counts in both a flat pool and per-condition
        paths_to_scan: List[Tuple[str, str, Path]] = []
        for sub in all_subjects:
            if sub in self._eo_paths:
                paths_to_scan.append((sub, "EO", self._eo_paths[sub]))
            if sub in self._ec_paths:
                paths_to_scan.append((sub, "EC", self._ec_paths[sub]))

        total_files = len(paths_to_scan)
        # Aggregate per-subject per-condition
        eo_map: Dict[str, np.ndarray] = {}
        ec_map: Dict[str, np.ndarray] = {}

        for i, (sub, cond, p) in enumerate(paths_to_scan):
            if self._abort:
                break
            lengths = _passages_from_path(p)
            if lengths is not None and lengths.size > 0:
                if cond == "EO":
                    eo_map[sub] = lengths
                else:
                    ec_map[sub] = lengths
            self.progress.emit(i + 1, total_files)

        self.finished.emit(eo_map, ec_map, n_total)


# ---------------------------------------------------------------------------
# "% data retained" and "participants retained" curve helpers
# ---------------------------------------------------------------------------

def _retention_curve(
    lengths: np.ndarray,
    window_sizes: np.ndarray,
    total_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    For each window size *w* (seconds), compute what fraction of the total
    recording duration could actually be analysed.

    A passage of length L contributes floor(L / w) * w usable seconds.
    If *total_seconds* is None, the sum of all passages is used as the
    normalisation denominator (i.e. assumes the full recording is the union
    of the passages — a lower bound).
    """
    total = float(np.sum(lengths)) if total_seconds is None else float(total_seconds)
    if total <= 0:
        return np.zeros_like(window_sizes, dtype=np.float64)

    pct = np.empty(len(window_sizes), dtype=np.float64)
    for j, w in enumerate(window_sizes):
        if w <= 0:
            pct[j] = 100.0
            continue
        usable = np.sum(np.floor(lengths / w) * w)
        pct[j] = 100.0 * float(usable) / total
    return pct


def _pct_participants_retained(
    per_condition: List[np.ndarray],
    n_total: int,
    window_sizes: np.ndarray,
) -> np.ndarray:
    """
    For each window size *w* (seconds), return the percentage of the total
    *n_total* participants who have at least one clean passage >= *w*.
    """
    if n_total == 0:
        return np.zeros(len(window_sizes), dtype=np.float64)
    pct = np.empty(len(window_sizes), dtype=np.float64)
    for j, w in enumerate(window_sizes):
        count = sum(1 for p in per_condition if np.any(p >= w))
        pct[j] = 100.0 * count / n_total
    return pct


def _pct_mdd_participants_retained(
    cond_map: Dict[str, np.ndarray],
    sub_info: Dict[str, str],
    window_sizes: np.ndarray,
) -> np.ndarray:
    """
    For each window size *w*, return the percentage of MDD-indicated subjects
    who have at least one clean passage >= *w*.

    Only counts subjects who ARE present in cond_map AND have indication=="MDD".
    Denom is total MDD subjects present in cond_map.
    """
    mdd_subs = [sid for sid, lengths in cond_map.items() if sub_info.get(sid) == "MDD"]
    n_total_mdd = len(mdd_subs)
    if n_total_mdd == 0:
        return np.zeros(len(window_sizes), dtype=np.float64)

    pct = np.empty(len(window_sizes), dtype=np.float64)
    for j, w in enumerate(window_sizes):
        count = sum(1 for sid in mdd_subs if np.any(cond_map[sid] >= w))
        pct[j] = 100.0 * count / n_total_mdd
    return pct


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------

class PassageHistView(ViewBase):
    """
    Two-tab view of artifact-free passage statistics.

    **Histogram tab** — distribution of passage lengths (s).
    **Retention tab** — two stacked plots:
      - % of total data retained vs analysis window size (s).
      - Number of participants retaining at least one epoch vs window size.
    """

    view_name = "Passage Hist"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Structured recording info (set by set_recordings_dict)
        self._eo_paths: Dict[str, Path] = {}   # {sub: path} for ses-1 EO
        self._ec_paths: Dict[str, Path] = {}   # {sub: path} for ses-1 EC
        self._n_total_subj: int = 0            # union of subjects across EO+EC
        self._sub_info: Dict[str, str] = {}    # normalized sid -> indication

        # scan results
        self._eo_lengths: Optional[np.ndarray] = None
        self._ec_lengths: Optional[np.ndarray] = None
        self._per_eo: Optional[List[np.ndarray]] = None
        self._per_ec: Optional[List[np.ndarray]] = None

        # Legacy placeholder for single-file view or current active set
        self._passage_lengths: Optional[np.ndarray] = None

        self._total_seconds_eo: Optional[float] = None
        self._total_seconds_ec: Optional[float] = None
        self._worker: Optional[_ScanWorker] = None
        self.fs: int = 1

        # Interactive slider state
        self._active_x: Optional[np.ndarray] = None
        self._active_y_data: Optional[np.ndarray] = None
        self._active_y_parts: Optional[np.ndarray] = None
        self._active_y_mdd: Optional[np.ndarray] = None

        layout = QtWidgets.QVBoxLayout(self)

        # ── Controls row ──────────────────────────────────────────────────
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)

        ctrl.addWidget(QtWidgets.QLabel("Bin width (s):"))
        self.spin_bin = QtWidgets.QDoubleSpinBox()
        self.spin_bin.setRange(0.1, 60.0)
        self.spin_bin.setDecimals(2)
        self.spin_bin.setSingleStep(0.5)
        self.spin_bin.setValue(1.0)
        ctrl.addWidget(self.spin_bin)

        ctrl.addWidget(QtWidgets.QLabel("X max (s):"))
        self.spin_xmax = QtWidgets.QDoubleSpinBox()
        self.spin_xmax.setRange(1.0, 3600.0)
        self.spin_xmax.setDecimals(1)
        self.spin_xmax.setSingleStep(5.0)
        self.spin_xmax.setValue(60.0)
        ctrl.addWidget(self.spin_xmax)

        ctrl.addSpacing(12)

        ctrl.addWidget(QtWidgets.QLabel("Toggle:"))
        self.combo_cond_toggle = QtWidgets.QComboBox()
        self.combo_cond_toggle.addItems(["EO", "EC"])
        ctrl.addWidget(self.combo_cond_toggle)

        ctrl.addSpacing(12)

        self.btn_scan_all = QtWidgets.QPushButton("Scan all recordings…")
        self.btn_scan_all.setToolTip(
            "Aggregate passage lengths across ALL recordings in the dataset.\n"
            "Per-file results are cached to a temp directory."
        )
        ctrl.addWidget(self.btn_scan_all)

        self.btn_clear_cache = QtWidgets.QPushButton("Clear cache")
        self.btn_clear_cache.setToolTip("Delete cached passage-length files from the temp directory.")
        ctrl.addWidget(self.btn_clear_cache)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        ctrl.addWidget(self.btn_cancel)

        ctrl.addStretch(1)

        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setStyleSheet("color: #586e75;")
        ctrl.addWidget(self.lbl_stats)

        # ── Progress bar ─────────────────────────────────────────────────
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # ── Tab widget with two plots ─────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # — Histogram tab —
        self.plot_hist = pg.PlotWidget()
        self.plot_hist.setBackground("w")
        self.plot_hist.showGrid(x=True, y=True, alpha=0.25)
        self.plot_hist.setLabel("bottom", "Passage length", units="s")
        self.plot_hist.setLabel("left", "Count")
        self.plot_hist.setMenuEnabled(False)
        self.tabs.addTab(self.plot_hist, "Histogram")

        # — Retention tab: two stacked plots in a GraphicsLayoutWidget —
        self._ret_glw = pg.GraphicsLayoutWidget()
        self._ret_glw.setBackground("w")

        # Upper plot: % data retained
        self.plot_ret = self._ret_glw.addPlot(row=0, col=0)
        self.plot_ret.showGrid(x=True, y=True, alpha=0.25)
        self.plot_ret.setLabel("left", "Data retained", units="%")
        self.plot_ret.setMenuEnabled(False)
        self.plot_ret.setYRange(0, 100, padding=0.02)
        self.plot_ret.hideAxis("bottom")  # shared x-axis: only show ticks on lower plot

        # Lower plot: % participants retained — EO and EC separately
        self.plot_parts = self._ret_glw.addPlot(row=1, col=0)
        self.plot_parts.showGrid(x=True, y=True, alpha=0.25)
        self.plot_parts.setLabel("bottom", "Window size", units="s")
        self.plot_parts.setLabel("left", "Participants retained", units="%")
        self.plot_parts.setMenuEnabled(False)
        self.plot_parts.setYRange(0, 100, padding=0.02)

        # Link x-axes so panning/zooming is synchronised
        self.plot_parts.setXLink(self.plot_ret)

        # Initialize legends once to avoid duplication
        self.plot_ret.addLegend(offset=(10, 5))
        self.plot_parts.addLegend(offset=(10, 5))

        self.tabs.addTab(self._ret_glw, "% Retained vs Window")

        # ── Interactive Slider (Vertical Line) ───────────────────────────
        self._v_slider = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=True,
            pen=pg.mkPen(color=(100, 100, 100, 200), width=1.5, style=QtCore.Qt.DashLine),
            hoverPen=pg.mkPen(color=(255, 0, 0, 255), width=2),
            label="",
        )
        self._v_slider.setVisible(False)
        self.plot_ret.addItem(self._v_slider)

        # Tooltip text
        self._v_slider_label = pg.TextItem(
            text="",
            color=(50, 50, 50),
            anchor=(0, 0),
            fill=(255, 255, 255, 220),
            border=(150, 150, 150),
        )
        self._v_slider_label.setVisible(False)
        self.plot_ret.addItem(self._v_slider_label)

        # Wiring
        self.spin_bin.valueChanged.connect(lambda _: self._refresh())
        self.spin_xmax.valueChanged.connect(lambda _: self._refresh())
        self.combo_cond_toggle.currentIndexChanged.connect(lambda _: self._refresh())
        self._v_slider.sigPositionChanged.connect(self._on_slider_moved)
        self.btn_scan_all.clicked.connect(self._start_scan)
        self.btn_cancel.clicked.connect(self._cancel_scan)
        self.btn_clear_cache.clicked.connect(self._clear_cache)
        self.tabs.currentChanged.connect(lambda _: self._refresh())

    # ── Public API ───────────────────────────────────────────────────────

    def set_recordings_dict(
        self,
        recordings: Dict[RecordingKey, Dict[str, Path]],
        sub_info: Dict[str, str],
    ) -> None:
        """
        Called by main_window whenever the recording dict is known/updated.

        Filters to ses-1 entries only and splits EO/EC into separate dicts
        keyed by subject ID so the scan worker can compute per-condition
        participant retention curves.
        """
        self._sub_info = sub_info
        eo: Dict[str, Path] = {}
        ec: Dict[str, Path] = {}
        for key, cond_dict in recordings.items():
            # Accept ses-1, ses-01, ses-1b etc. — anything starting with "ses-1"
            if not key.ses.lower().startswith("ses-1"):
                continue
            if "EO" in cond_dict:
                eo[key.sub] = cond_dict["EO"]
            if "EC" in cond_dict:
                ec[key.sub] = cond_dict["EC"]
        self._eo_paths = eo
        self._ec_paths = ec
        self._n_total_subj = len(set(eo) | set(ec))

    def set_recording(
        self,
        eeg_path: Optional[Path],
        eeg: Dict[str, Any],
        data: np.ndarray,
        labels: List[str],
        fs: int,
        artifact_model: ArtifactModel,
    ) -> None:
        """Show passages for the currently loaded file (single-file view)."""
        self.fs = fs
        self._total_seconds = data.shape[1] / float(fs) if data is not None else None
        self._passage_lengths = self._compute_passages_from_artifact_model(
            data, fs, artifact_model
        )
        # Single-file mode: EO/EC participant curves are not meaningful
        self._per_eo = None
        self._per_ec = None
        if self._passage_lengths is not None and self._passage_lengths.size > 0:
            p99 = float(np.percentile(self._passage_lengths, 99))
            self.spin_xmax.blockSignals(True)
            self.spin_xmax.setValue(max(1.0, round(p99, 1)))
            self.spin_xmax.blockSignals(False)
        fname = eeg_path.name if eeg_path else "current file"
        self.lbl_stats.setText(f"Showing: {fname}")
        self._refresh()

    # ── Computation ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_passages_from_artifact_model(
        data: np.ndarray,
        fs: int,
        artifact_model: ArtifactModel,
    ) -> np.ndarray:
        """Use ArtifactModel.artifact_mask — identical pattern to pcorr.py."""
        n_samp = data.shape[1]
        if artifact_model.artifact_mask is not None:
            clean = ~artifact_model.artifact_mask
            segs = mask_to_segments(clean)
        else:
            segs = np.array([[0, n_samp]], dtype=np.int64)
        if segs.size == 0:
            return np.array([], dtype=np.float64)
        return (segs[:, 1] - segs[:, 0]).astype(np.float64) / float(fs)

    # ── Scan-all background job ───────────────────────────────────────────

    def _start_scan(self) -> None:
        if self._worker is not None:
            return
        if not (self._eo_paths or self._ec_paths):
            QtWidgets.QMessageBox.information(
                self,
                "No recordings",
                "No ses-1 recording paths are known. Load a dataset first.",
            )
            return

        n_files = len(self._eo_paths) + len(self._ec_paths)
        self.btn_scan_all.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, n_files)
        self.progress_bar.setValue(0)
        self.lbl_stats.setText(f"Scanning 0 / {n_files} files (ses-1 EO+EC)…")

        self._worker = _ScanWorker(self._eo_paths, self._ec_paths)
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    def _cancel_scan(self) -> None:
        if self._worker:
            self._worker.abort()

    def _on_scan_progress(self, done: int, total: int) -> None:
        self.progress_bar.setValue(done)
        self.lbl_stats.setText(f"Scanning {done} / {total} files…")

    def _on_scan_finished(
        self,
        eo_map: Dict[str, np.ndarray],
        ec_map: Dict[str, np.ndarray],
        n_total: int,
    ) -> None:
        self._worker = None
        self.btn_scan_all.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setVisible(False)

        # Store maps
        self._per_eo_map = eo_map
        self._per_ec_map = ec_map
        self._n_total_subj = n_total

        # Compute combined arrays and lists for display
        self._per_eo = list(eo_map.values())
        self._per_ec = list(ec_map.values())
        self._eo_lengths = np.concatenate(self._per_eo) if self._per_eo else np.array([], dtype=np.float64)
        self._ec_lengths = np.concatenate(self._per_ec) if self._per_ec else np.array([], dtype=np.float64)

        # Set default active set based on toggle
        cond = self.combo_cond_toggle.currentText()
        self._passage_lengths = self._eo_lengths if cond == "EO" else self._ec_lengths
        self._total_seconds_eo = None  # aggregation denominators
        self._total_seconds_ec = None

        if self._passage_lengths is not None and self._passage_lengths.size > 0:
            p99 = float(np.percentile(self._passage_lengths, 99))
            self.spin_xmax.blockSignals(True)
            self.spin_xmax.setValue(max(1.0, round(p99, 1)))
            self.spin_xmax.blockSignals(False)
            self.lbl_stats.setText(
                f"ses-1 · {n_total} subjects · "
                f"{len(eo_map)} with EO · {len(ec_map)} with EC"
            )
        else:
            self.lbl_stats.setText("No clean passages found in ses-1 recordings.")
        self._refresh()

    def _clear_cache(self) -> None:
        _TMP_DIR.mkdir(parents=True, exist_ok=True)
        removed = 0
        for f in _TMP_DIR.glob("*.npy"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
        QtWidgets.QMessageBox.information(
            self, "Cache cleared", f"Removed {removed} cached file(s) from {_TMP_DIR}."
        )

    # ── Display ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        tab = self.tabs.currentIndex()
        if tab == 0:
            self._refresh_histogram()
        else:
            self._refresh_retention()

    def _refresh_histogram(self) -> None:
        self.plot_hist.clear()

        if self._passage_lengths is None or self._passage_lengths.size == 0:
            self.lbl_stats.setText(
                self.lbl_stats.text() or "No data — load a file or run 'Scan all'."
            )
            return

        lengths = self._passage_lengths
        bin_width = float(self.spin_bin.value())
        xmax = float(self.spin_xmax.value())

        bins = np.arange(0.0, xmax + bin_width, bin_width)
        counts, edges = np.histogram(lengths, bins=bins)

        bar = pg.BarGraphItem(
            x=edges[:-1],
            height=counts,
            width=bin_width * 0.9,
            brush=pg.mkBrush(70, 130, 210, 180),
            pen=pg.mkPen("w", width=0.5),
        )
        self.plot_hist.addItem(bar)

        med = float(np.median(lengths))
        median_line = pg.InfiniteLine(
            pos=med,
            angle=90,
            movable=False,
            pen=pg.mkPen(color=(220, 80, 40), width=2, style=QtCore.Qt.DashLine),
            label=f"median {med:.2f}s",
            labelOpts={
                "position": 0.85,
                "color": (220, 80, 40),
                "fill": (255, 255, 255, 150),
            },
        )
        self.plot_hist.addItem(median_line)

        self.plot_hist.setXRange(0.0, xmax, padding=0.01)
        if counts.max() > 0:
            self.plot_hist.setYRange(0, int(counts.max() * 1.1), padding=0.0)

        self._update_stats_label(lengths, xmax)

    def _refresh_retention(self) -> None:
        self.plot_ret.clear()
        self.plot_parts.clear()

        # Re-add legend and slider since clear() removes them
        self.plot_ret.addLegend(offset=(10, 5))
        self.plot_ret.addItem(self._v_slider)
        self.plot_ret.addItem(self._v_slider_label)

        cond = self.combo_cond_toggle.currentText()
        if cond == "EO":
            lengths = self._eo_lengths
            per_list = self._per_eo
            per_map = getattr(self, "_per_eo_map", {})
            total_sec = getattr(self, "_total_seconds_eo", None)
        else:
            lengths = self._ec_lengths
            per_list = self._per_ec
            per_map = getattr(self, "_per_ec_map", {})
            total_sec = getattr(self, "_total_seconds_ec", None)

        # Single-file fallback
        if lengths is None or lengths.size == 0:
            if self._passage_lengths is not None and self._passage_lengths.size > 0:
                lengths = self._passage_lengths
                total_sec = getattr(self, "_total_seconds", None)
            else:
                return

        xmax = float(self.spin_xmax.value())
        n_pts = 120
        window_sizes = np.linspace(0.1, xmax, n_pts)

        # ── Data retained (%) (Blue) ──────────────────────────────────
        pct_data = _retention_curve(lengths, window_sizes, total_seconds=total_sec)
        self.plot_ret.plot(
            window_sizes, pct_data,
            pen=pg.mkPen(color=(70, 130, 210), width=3),
            name="Data retained (%)"
        )

        # ── Participant curves (if multi-file) ───────────────────────
        prefix = ""
        if per_list:
            n_total_active = len(per_list)
            # Count MDD subjects specifically in THIS condition (per_map)
            n_mdd_active = sum(1 for sid in per_map if self._sub_info.get(sid) == "MDD")

            prefix = f"N={n_total_active} subjects (MDD={n_mdd_active})"

            # 1. All participants (%) (Orange Dash)
            pct_parts = _pct_participants_retained(per_list, self._n_total_subj or 1, window_sizes)
            self.plot_ret.plot(
                window_sizes, pct_parts,
                pen=pg.mkPen(color=(230, 140, 30), width=2, style=QtCore.Qt.DashLine),
                name="All participants",
            )

            # 2. MDD participants (%) (Purple)
            if self._sub_info:
                pct_mdd = _pct_mdd_participants_retained(per_map, self._sub_info, window_sizes)
                self.plot_ret.plot(
                    window_sizes, pct_mdd,
                    pen=pg.mkPen(color=(160, 80, 200), width=2),
                    name="MDD participants",
                )

        # Mark 80 % and 90 % thresholds
        for threshold, color in [(90.0, (40, 180, 80)), (80.0, (220, 160, 40))]:
            line = pg.InfiniteLine(
                pos=threshold, angle=0, movable=False,
                pen=pg.mkPen(color=color, width=1, style=QtCore.Qt.DashLine),
                label=f"{threshold:.0f}%",
                labelOpts={"position": 0.05, "color": color},
            )
            self.plot_ret.addItem(line)

        self.plot_ret.setXRange(0.0, xmax, padding=0.01)
        self.plot_ret.setYRange(0, 100, padding=0.02)
        self.plot_ret.showAxis("bottom")
        self.plot_ret.setLabel("bottom", "Window size", units="s")

        # Hide the lower plot as we've merged everything into plot_ret
        self.plot_parts.setVisible(False)

        # ── Update Slider State ──
        self._active_x = window_sizes
        self._active_y_data = pct_data
        if per_list:
            self._active_y_parts = pct_parts
            if self._sub_info:
                self._active_y_mdd = pct_mdd
            else:
                self._active_y_mdd = None
        else:
            self._active_y_parts = None
            self._active_y_mdd = None

        # Reset/Show slider at a reasonable starting point if hidden
        if not self._v_slider.isVisible():
            self._v_slider.setPos(xmax * 0.1)
            self._v_slider.setVisible(True)
            self._v_slider_label.setVisible(True)

        self._on_slider_moved()
        self._update_stats_label(lengths, xmax, prefix_override=prefix)

    def _on_slider_moved(self) -> None:
        if self._active_x is None:
            return

        x = self._v_slider.value()
        # Interpolate values
        y_data = np.interp(x, self._active_x, self._active_y_data)

        lines = [f"Window: {x:.1f} s", f"Data: {y_data:.1f} %"]

        if self._active_y_parts is not None:
            y_parts = np.interp(x, self._active_x, self._active_y_parts)
            lines.append(f"All Parts: {y_parts:.1f} %")

        if self._active_y_mdd is not None:
            y_mdd = np.interp(x, self._active_x, self._active_y_mdd)
            lines.append(f"MDD Parts: {y_mdd:.1f} %")

        self._v_slider_label.setText("\n".join(lines))

        # Position label relative to slider
        # offset slightly to the right of the line
        self._v_slider_label.setPos(x, 95)  # fixed y topish

    def _update_stats_label(
        self, lengths: np.ndarray, xmax: float, prefix_override: Optional[str] = None
    ) -> None:
        n_total = lengths.size
        n_shown = int(np.sum(lengths <= xmax))
        mean_s = float(np.mean(lengths))
        med = float(np.median(lengths))
        p25, p75 = np.percentile(lengths, [25, 75])
        extra = f" · {n_total - n_shown} beyond Xmax" if n_shown < n_total else ""

        if prefix_override:
            base = prefix_override
        else:
            # preserve any prefix text (e.g. "Showing: foo.npy")
            prefix = self.lbl_stats.text().split("·")[0].strip()
            # rebuild keeping only the filename/count prefix
            if prefix.startswith("Showing") or prefix.startswith("All") or prefix.startswith("N="):
                base = prefix
            else:
                base = ""

        stats = (
            f"{n_total} passages · mean {mean_s:.2f}s · median {med:.2f}s"
            f" · IQR [{p25:.2f}–{p75:.2f}]s{extra}"
        )
        self.lbl_stats.setText(f"{base}  ·  {stats}" if base else stats)
