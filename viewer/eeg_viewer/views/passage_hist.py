from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg

from ..qt_compat import QtCore, QtWidgets
from ..artifacts import ArtifactModel, mask_to_segments
from ..io import load_eeg_dict
from .base import ViewBase


def _passages_from_path(path: Path) -> Optional[np.ndarray]:
    """
    Load a single .npy file and return its artifact-free passage lengths (seconds).
    Returns None on load error. Returns a zero-element array if fully artifacted.
    """
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

    # Find artifact channel
    artifact_idx: Optional[int] = None
    for i, lab in enumerate(labels):
        if str(lab).strip().lower() in {"artifact", "artifacts", "artfct", "art"}:
            artifact_idx = i
            break

    if artifact_idx is not None:
        art_ch = data[artifact_idx]
        clean_mask = art_ch == 0
        segs = mask_to_segments(clean_mask)
    else:
        segs = np.array([[0, n_samp]], dtype=np.int64)

    if segs.size == 0:
        return np.array([], dtype=np.float64)

    lengths_s = (segs[:, 1] - segs[:, 0]).astype(np.float64) / float(fs)
    return lengths_s


class _ScanWorker(QtCore.QThread):
    """Background thread: scans all paths and accumulates passage lengths."""

    progress = QtCore.Signal(int, int)          # (done, total)
    finished = QtCore.Signal(object)            # np.ndarray of all lengths

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

        combined = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.float64)
        self.finished.emit(combined)


class PassageHistView(ViewBase):
    """
    Histogram of artifact-free passage lengths (in seconds).

    By default shows passages for the currently loaded file.
    Click "Scan all" to aggregate passages across every recording in the dataset.
    """

    view_name = "Passage Hist"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # All paths known to the viewer (set by main_window via set_all_paths())
        self._all_paths: List[Path] = []
        self._passage_lengths: Optional[np.ndarray] = None
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
            "Aggregate passage lengths across ALL recordings in the dataset."
        )
        ctrl.addWidget(self.btn_scan_all)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        ctrl.addWidget(self.btn_cancel)

        ctrl.addStretch(1)

        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setStyleSheet("color: #586e75;")
        ctrl.addWidget(self.lbl_stats)

        # ── Progress bar (hidden until scan) ─────────────────────────────
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # ── Plot ─────────────────────────────────────────────────────────
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Passage length", units="s")
        self.plot.setLabel("left", "Count")
        self.plot.setMenuEnabled(False)
        layout.addWidget(self.plot, 1)

        # Wiring
        self.spin_bin.valueChanged.connect(lambda _: self._refresh())
        self.spin_xmax.valueChanged.connect(lambda _: self._refresh())
        self.btn_scan_all.clicked.connect(self._start_scan)
        self.btn_cancel.clicked.connect(self._cancel_scan)

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

    # ── Display ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self.plot.clear()

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
        self.plot.addItem(bar)

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
        self.plot.addItem(median_line)

        self.plot.setXRange(0.0, xmax, padding=0.01)
        if counts.max() > 0:
            self.plot.setYRange(0, int(counts.max() * 1.1), padding=0.0)

        # Stats
        n_total = lengths.size
        n_shown = int(np.sum(lengths <= xmax))
        mean_s = float(np.mean(lengths))
        p25, p75 = np.percentile(lengths, [25, 75])
        extra = f" · {n_total - n_shown} beyond Xmax" if n_shown < n_total else ""
        self.lbl_stats.setText(
            f"{n_total} passages · mean {mean_s:.2f}s · median {med:.2f}s"
            f" · IQR [{p25:.2f}–{p75:.2f}]s{extra}"
        )
