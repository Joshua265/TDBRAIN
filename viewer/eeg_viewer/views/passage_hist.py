from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg

from ..qt_compat import QtCore, QtWidgets
from ..artifacts import ArtifactModel, mask_to_segments
from ..io import load_eeg_dict
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
    """Background thread: scans all paths, uses cache where available."""

    progress = QtCore.Signal(int, int)   # (done, total)
    finished = QtCore.Signal(object)     # np.ndarray of all lengths

    def __init__(self, paths: List[Path], parent=None):
        super().__init__(parent)
        self._paths = paths
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        all_lengths: List[np.ndarray] = []
        total = len(self._paths)
        for i, p in enumerate(self._paths):
            if self._abort:
                break
            lengths = _passages_from_path(p)
            if lengths is not None and lengths.size > 0:
                all_lengths.append(lengths)
            self.progress.emit(i + 1, total)

        combined = (
            np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.float64)
        )
        self.finished.emit(combined)


# ---------------------------------------------------------------------------
# "% data retained" curve helpers
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


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------

class PassageHistView(ViewBase):
    """
    Two-tab view of artifact-free passage statistics.

    **Histogram tab** — distribution of passage lengths (s).
    **Retention tab** — % of total data retained as a function of analysis
    window size (s).  Useful for choosing an epoch length that maximises
    usable data.
    """

    view_name = "Passage Hist"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._all_paths: List[Path] = []
        self._passage_lengths: Optional[np.ndarray] = None
        self._total_seconds: Optional[float] = None   # full recording duration
        self._worker: Optional[_ScanWorker] = None
        self.fs: int = 1

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

        # — Retention tab —
        self.plot_ret = pg.PlotWidget()
        self.plot_ret.setBackground("w")
        self.plot_ret.showGrid(x=True, y=True, alpha=0.25)
        self.plot_ret.setLabel("bottom", "Window size", units="s")
        self.plot_ret.setLabel("left", "Data retained", units="%")
        self.plot_ret.setMenuEnabled(False)
        self.plot_ret.setYRange(0, 100, padding=0.02)
        self.tabs.addTab(self.plot_ret, "% Retained vs Window")

        # Wiring
        self.spin_bin.valueChanged.connect(lambda _: self._refresh())
        self.spin_xmax.valueChanged.connect(lambda _: self._refresh())
        self.btn_scan_all.clicked.connect(self._start_scan)
        self.btn_cancel.clicked.connect(self._cancel_scan)
        self.btn_clear_cache.clicked.connect(self._clear_cache)
        self.tabs.currentChanged.connect(lambda _: self._refresh())

    # ── Public API ───────────────────────────────────────────────────────

    def set_all_paths(self, paths: List[Path]) -> None:
        """Called by main_window whenever the recording list is known/updated."""
        self._all_paths = list(paths)

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
        if not self._all_paths:
            QtWidgets.QMessageBox.information(
                self, "No recordings", "No recording paths are known. Load a dataset first."
            )
            return

        self.btn_scan_all.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self._all_paths))
        self.progress_bar.setValue(0)
        self.lbl_stats.setText(f"Scanning 0 / {len(self._all_paths)} files…")

        self._worker = _ScanWorker(self._all_paths)
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    def _cancel_scan(self) -> None:
        if self._worker:
            self._worker.abort()

    def _on_scan_progress(self, done: int, total: int) -> None:
        self.progress_bar.setValue(done)
        self.lbl_stats.setText(f"Scanning {done} / {total} files…")

    def _on_scan_finished(self, lengths: np.ndarray) -> None:
        self._worker = None
        self.btn_scan_all.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress_bar.setVisible(False)
        self._total_seconds = None  # use sum of passages as denominator for all-file view

        self._passage_lengths = lengths
        if lengths.size > 0:
            p99 = float(np.percentile(lengths, 99))
            self.spin_xmax.blockSignals(True)
            self.spin_xmax.setValue(max(1.0, round(p99, 1)))
            self.spin_xmax.blockSignals(False)
            n_files = len(self._all_paths)
            self.lbl_stats.setText(f"All {n_files} recordings")
        else:
            self.lbl_stats.setText("No clean passages found across all recordings.")
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

        if self._passage_lengths is None or self._passage_lengths.size == 0:
            return

        lengths = self._passage_lengths
        xmax = float(self.spin_xmax.value())

        # Window sizes: 30 steps from 0.1 s up to xmax
        n_pts = 120
        window_sizes = np.linspace(0.1, xmax, n_pts)
        pct = _retention_curve(lengths, window_sizes, total_seconds=self._total_seconds)

        curve = self.plot_ret.plot(
            window_sizes,
            pct,
            pen=pg.mkPen(color=(70, 130, 210), width=2),
        )

        # Mark 80 % and 90 % thresholds
        for threshold, color in [(90.0, (40, 180, 80)), (80.0, (220, 160, 40))]:
            line = pg.InfiniteLine(
                pos=threshold,
                angle=0,
                movable=False,
                pen=pg.mkPen(color=color, width=1, style=QtCore.Qt.DashLine),
                label=f"{threshold:.0f}%",
                labelOpts={"position": 0.05, "color": color},
            )
            self.plot_ret.addItem(line)

        self.plot_ret.setXRange(0.0, xmax, padding=0.01)
        self.plot_ret.setYRange(0, 100, padding=0.02)

        self._update_stats_label(lengths, xmax)

    def _update_stats_label(self, lengths: np.ndarray, xmax: float) -> None:
        n_total = lengths.size
        n_shown = int(np.sum(lengths <= xmax))
        mean_s = float(np.mean(lengths))
        med = float(np.median(lengths))
        p25, p75 = np.percentile(lengths, [25, 75])
        extra = f" · {n_total - n_shown} beyond Xmax" if n_shown < n_total else ""
        # preserve any prefix text (e.g. "Showing: foo.npy")
        prefix = self.lbl_stats.text().split("·")[0].strip()
        # rebuild keeping only the filename/count prefix
        if prefix.startswith("Showing") or prefix.startswith("All"):
            base = prefix
        else:
            base = ""
        stats = (
            f"{n_total} passages · mean {mean_s:.2f}s · median {med:.2f}s"
            f" · IQR [{p25:.2f}–{p75:.2f}]s{extra}"
        )
        self.lbl_stats.setText(f"{base}  {stats}" if base else stats)
