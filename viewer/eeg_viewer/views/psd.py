from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from ..qt_compat import QtCore, QtWidgets
from ..artifacts import ArtifactModel
from .base import ViewBase


def welch_psd_numpy(
    x: np.ndarray,
    fs: int,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal Welch PSD implementation (SciPy-free).
    Returns (freqs, psd) where psd is one-sided power spectral density.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 8:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    nperseg = int(min(max(8, nperseg), n))
    noverlap = int(min(max(0, noverlap), nperseg - 1))
    step = nperseg - noverlap
    if step <= 0:
        step = nperseg

    w = np.hanning(nperseg).astype(np.float64)
    w2 = float(np.sum(w * w)) or 1.0

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / float(fs))
    pxx_acc = np.zeros(freqs.shape, dtype=np.float64)
    nseg = 0

    for start in range(0, n - nperseg + 1, step):
        seg = x[start : start + nperseg]
        seg = seg - np.mean(seg)
        segw = seg * w
        X = np.fft.rfft(segw)
        P = (np.abs(X) ** 2) / (float(fs) * w2)

        # one-sided correction: double bins except DC and Nyquist (if present)
        if P.size > 2:
            P[1:-1] *= 2.0
        elif P.size == 2:
            P[1] *= 2.0

        pxx_acc += P
        nseg += 1

    if nseg == 0:
        return freqs, np.zeros_like(freqs)

    return freqs, pxx_acc / float(nseg)


class PSDView(ViewBase):
    view_name = "PSD"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Selected channel", "Average EEG-only"])
        self.mode.setToolTip(
            "Choose whether to show the selected channel PSD or the mean PSD across EEG channels."
        )
        ctrl.addWidget(QtWidgets.QLabel("Mode:"))
        ctrl.addWidget(self.mode)

        self.combo_channel = QtWidgets.QComboBox()
        self.combo_channel.setMinimumWidth(220)
        ctrl.addWidget(QtWidgets.QLabel("Channel:"))
        ctrl.addWidget(self.combo_channel)

        self.chk_use_visible = QtWidgets.QCheckBox("Use visible window")
        self.chk_use_visible.setChecked(True)
        self.chk_use_visible.setToolTip(
            "Compute PSD from the time window currently visible in the time-series view."
        )
        ctrl.addWidget(self.chk_use_visible)

        self.chk_no_artifact = QtWidgets.QCheckBox("Artifact=0 only")
        self.chk_no_artifact.setChecked(False)
        self.chk_no_artifact.setToolTip(
            "Exclude samples where the artifact channel is non-zero before computing the PSD."
        )
        ctrl.addWidget(self.chk_no_artifact)

        self.chk_db = QtWidgets.QCheckBox("dB")
        self.chk_db.setChecked(True)
        ctrl.addWidget(self.chk_db)

        ctrl.addStretch(1)

        self.spin_fmax = QtWidgets.QDoubleSpinBox()
        self.spin_fmax.setRange(5.0, 250.0)
        self.spin_fmax.setValue(80.0)  # default per your hardware note
        self.spin_fmax.setSingleStep(5.0)
        ctrl.addWidget(QtWidgets.QLabel("Fmax:"))
        ctrl.addWidget(self.spin_fmax)

        # Guide lines toggles
        self.chk_50 = QtWidgets.QCheckBox("50 Hz")
        self.chk_50.setChecked(True)
        self.chk_80 = QtWidgets.QCheckBox("80 Hz")
        self.chk_80.setChecked(True)
        self.chk_100 = QtWidgets.QCheckBox("100 Hz")
        self.chk_100.setChecked(False)
        ctrl.addWidget(self.chk_50)
        ctrl.addWidget(self.chk_80)
        ctrl.addWidget(self.chk_100)

        # Warning line
        self.lbl_warn = QtWidgets.QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color: #b58900;")  # muted warning tone
        layout.addWidget(self.lbl_warn)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Power", units="dB")
        self.plot.setMenuEnabled(False)
        layout.addWidget(self.plot, 1)

        self.curve = self.plot.plot([], [])

        # guide lines
        self.line_50 = pg.InfiniteLine(pos=50, angle=90, movable=False)
        self.line_80 = pg.InfiniteLine(pos=80, angle=90, movable=False)
        self.line_100 = pg.InfiniteLine(pos=100, angle=90, movable=False)
        self.plot.addItem(self.line_50)
        self.plot.addItem(self.line_80)
        self.plot.addItem(self.line_100)

        # Data
        self.eeg_path: Optional[Path] = None
        self.data: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.fs: int = 1
        self.artifact_model: Optional[ArtifactModel] = None

        self._i0: int = 0
        self._i1: Optional[int] = None

        # Wiring
        self.mode.currentIndexChanged.connect(lambda _: self._on_mode_changed())
        self.combo_channel.currentIndexChanged.connect(lambda _: self.update_psd())
        self.chk_use_visible.toggled.connect(lambda _: self.update_psd())
        self.chk_no_artifact.toggled.connect(lambda _: self.update_psd())
        self.chk_db.toggled.connect(lambda _: self.update_psd())
        self.spin_fmax.valueChanged.connect(lambda _: self.update_psd())
        self.chk_50.toggled.connect(lambda _: self._update_guides())
        self.chk_80.toggled.connect(lambda _: self._update_guides())
        self.chk_100.toggled.connect(lambda _: self._update_guides())

        self._on_mode_changed()
        self._update_guides()

    def set_recording(
        self,
        eeg_path: Optional[Path],
        eeg: Dict[str, Any],
        data: np.ndarray,
        labels: List[str],
        fs: int,
        artifact_model: ArtifactModel,
    ) -> None:
        self.eeg_path = eeg_path
        self.data = data
        self.labels = labels
        self.fs = fs
        self.artifact_model = artifact_model
        self._i0 = 0
        self._i1 = data.shape[1] if data is not None else None

        self.combo_channel.blockSignals(True)
        self.combo_channel.clear()
        for i, lab in enumerate(labels):
            self.combo_channel.addItem(str(lab), userData=int(i))
        self.combo_channel.setCurrentIndex(0)
        self.combo_channel.blockSignals(False)

        self.update_psd()

    def set_visible_window_samples(self, i0: int, i1: int) -> None:
        self._i0 = int(i0)
        self._i1 = int(i1)
        if self.chk_use_visible.isChecked():
            self.update_psd()

    def _on_mode_changed(self):
        # If averaging, channel picker is informational only; make it obvious.
        is_avg = self.mode.currentText() == "Average EEG-only"
        self.combo_channel.setEnabled(not is_avg)
        self.update_psd()

    def _update_guides(self):
        self.line_50.setVisible(self.chk_50.isChecked())
        self.line_80.setVisible(self.chk_80.isChecked())
        self.line_100.setVisible(self.chk_100.isChecked())

    @staticmethod
    def _is_aux_channel(label: str) -> bool:
        s = label.strip().lower()
        aux_exact = {
            "veog",
            "heog",
            "ecg",
            "erbs",
            "orbocc",
            "mass",
            "artifacts",
            "artifact",
        }
        if s in aux_exact:
            return True
        if "eog" in s or "ecg" in s or "emg" in s:
            return True
        if "artifact" in s:
            return True
        return False

    def _eeg_channel_indices(self) -> List[int]:
        return [
            i for i, lab in enumerate(self.labels) if not self._is_aux_channel(str(lab))
        ]

    def _selected_channel_index(self) -> Optional[int]:
        # Robust channel extraction (fixes “doesn’t change” issues across bindings)
        idx = self.combo_channel.currentIndex()
        if idx < 0:
            return None
        v = self.combo_channel.itemData(idx)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _artifact_mask_in_window(self, i0: int, i1: int) -> Optional[np.ndarray]:
        """Return boolean mask (True = clean sample) for samples i0..i1, or None."""
        if (
            not self.chk_no_artifact.isChecked()
            or self.artifact_model is None
            or self.artifact_model.artifact_mask is None
        ):
            return None
        mask = self.artifact_model.artifact_mask[i0:i1]  # True where artifact != 0
        clean = ~mask  # True where artifact == 0
        return clean

    def update_psd(self) -> None:
        if self.data is None or self.data.size == 0:
            self.curve.setData([], [])
            return

        n_ch, n_samp = self.data.shape

        # Window selection
        i0, i1 = 0, n_samp
        if self.chk_use_visible.isChecked() and self._i1 is not None:
            i0 = max(0, min(n_samp - 2, self._i0))
            i1 = max(i0 + 2, min(n_samp, self._i1))

        # Artifact masking: keep only clean samples
        clean_mask = self._artifact_mask_in_window(i0, i1)

        def _extract(ch_idx: int) -> np.ndarray:
            """Return the (possibly masked) 1-D signal for channel ch_idx."""
            x = self.data[ch_idx, i0:i1].astype(np.float64, copy=False)  # type: ignore[index]
            if clean_mask is not None:
                x = x[clean_mask]
            return x

        seg_len = int(clean_mask.sum()) if clean_mask is not None else (i1 - i0)
        if seg_len < 16:
            self.curve.setData([], [])
            return

        # Welch params: ~2 second windows
        nperseg = int(min(seg_len, max(256, int(self.fs * 2))))
        noverlap = int(nperseg // 2)

        # Compute PSD
        if self.mode.currentText() == "Average EEG-only":
            chs = self._eeg_channel_indices()
            if not chs:
                self.curve.setData([], [])
                return
            psds = []
            f_ref = None
            for ch in chs:
                x = _extract(ch)
                f, p = welch_psd_numpy(x, self.fs, nperseg=nperseg, noverlap=noverlap)
                if p.size:
                    f_ref = f
                    psds.append(p)
            if not psds or f_ref is None:
                self.curve.setData([], [])
                return
            f = f_ref
            pxx = np.mean(np.vstack(psds), axis=0)
        else:
            ch = self._selected_channel_index()
            if ch is None or ch < 0 or ch >= n_ch:
                self.curve.setData([], [])
                return
            x = _extract(ch)
            f, pxx = welch_psd_numpy(x, self.fs, nperseg=nperseg, noverlap=noverlap)

        if f.size == 0 or pxx.size == 0:
            self.curve.setData([], [])
            return

        # Hardware interpretation warning
        fmax = float(self.spin_fmax.value())
        if fmax > 80.0:
            self.lbl_warn.setText(
                "Note: >~80 Hz may show roll-off from the acquisition low-pass (~100 Hz). Interpret cautiously."
            )
        else:
            self.lbl_warn.setText("")

        keep = f <= fmax
        f = f[keep]
        pxx = pxx[keep]

        if self.chk_db.isChecked():
            y = 10.0 * np.log10(np.maximum(pxx, 1e-20))
            self.plot.setLabel("left", "Power", units="dB")
        else:
            y = pxx
            self.plot.setLabel("left", "Power", units="µV²/Hz")

        self.curve.setData(f, y)
        self.plot.setXRange(0.0, fmax, padding=0.02)
