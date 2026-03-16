from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from ..qt_compat import QtCore, QtGui, QtWidgets
from ..artifacts import ArtifactModel
from .base import ViewBase
from .psd import welch_psd_numpy

# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------
BANDS: List[Tuple[str, float, float]] = [
    ("Delta",      0.5,  3.0),
    ("Theta",      4.0,  7.0),
    ("Alpha",      8.0, 12.0),
    ("Beta-low",  12.0, 15.0),
    ("Beta-mid",  16.0, 20.0),
    ("Beta-high", 21.0, 30.0),
    ("Gamma-low", 30.0, 40.0),
    ("Gamma-high",60.0, 80.0),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUX_EXACT = {
    "veog", "heog", "ecg", "erbs", "orbocc", "mass", "artifacts", "artifact",
}


def _is_aux(label: str) -> bool:
    s = label.strip().lower()
    if s in _AUX_EXACT:
        return True
    if "eog" in s or "ecg" in s or "emg" in s or "artifact" in s:
        return True
    return False


def _band_power(freqs: np.ndarray, pxx: np.ndarray, flo: float, fhi: float) -> float:
    """Integrate PSD between flo and fhi (trapezoidal rule)."""
    mask = (freqs >= flo) & (freqs <= fhi)
    if mask.sum() < 1:
        return 0.0
    return float(np.trapz(pxx[mask], freqs[mask]))


def _corr_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrix from an (n_channels,) 1-D vector
    stored as rows of a (n_channels,) array – actually computes it
    from an (n_channels, n_samples) matrix where each row is a channel
    time-series, OR just from an (n_channels,) band-power vector
    by computing scalar correlations.

    Here *mat* is shape (n_channels, n_windows) where each column is
    the band-power of each channel for one PSD window. Pearson corr.
    is computed across windows.  If n_windows == 1 we use the band-power
    vector directly and compute channel×channel correlation with a trick.
    """
    n_ch, n_win = mat.shape
    if n_win < 2:
        # Can't compute a proper correlation; return identity.
        return np.eye(n_ch)
    # Mean-center each channel's window series, then correlate.
    m = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    # Avoid divide-by-zero
    norms = np.where(norms == 0, 1.0, norms)
    m_norm = m / norms
    R = m_norm @ m_norm.T
    return np.clip(R, -1.0, 1.0)


def _compute_band_corr(
    data: np.ndarray,
    labels: List[str],
    fs: int,
    i0: int,
    i1: int,
    band_lo: float,
    band_hi: float,
    eeg_only: bool,
    clean_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (corr_matrix [n_ch × n_ch], channel_labels).

    Strategy: Welch with short (1-s) segments so we get multiple power
    windows → use per-segment band-power as the variable to correlate.
    """
    n_ch, n_samp = data.shape

    chs = list(range(n_ch))
    ch_labels = labels
    if eeg_only:
        chs = [i for i, lab in enumerate(labels) if not _is_aux(str(lab))]
        ch_labels = [labels[i] for i in chs]

    if not chs:
        return np.empty((0, 0)), []

    seg_len = i1 - i0
    if seg_len < 32:
        n = len(chs)
        return np.eye(n), ch_labels

    # Apply artifact mask: keep only clean samples inside the window
    raw = data[:, i0:i1]
    if clean_mask is not None and clean_mask.size == seg_len:
        raw = raw[:, clean_mask]
    seg_len_used = raw.shape[1]
    if seg_len_used < 32:
        n = len(chs)
        return np.eye(n), ch_labels

    # Use ~2 s Welch segments (like PSDView) but collect per-segment power.
    nperseg = int(min(seg_len_used, max(256, int(fs * 2))))
    noverlap = int(nperseg // 2)
    step = nperseg - noverlap
    if step <= 0:
        step = nperseg

    # Build per-channel, per-segment band power matrix
    band_powers: List[np.ndarray] = []
    for ch_idx in chs:
        x = raw[ch_idx].astype(np.float64, copy=False)
        seg_powers: List[float] = []

        w = np.hanning(nperseg)
        w2 = float(np.sum(w * w)) or 1.0
        freqs_seg = np.fft.rfftfreq(nperseg, d=1.0 / float(fs))

        for start in range(0, len(x) - nperseg + 1, step):
            seg = x[start: start + nperseg]
            seg = seg - seg.mean()
            segw = seg * w
            X = np.fft.rfft(segw)
            P = (np.abs(X) ** 2) / (float(fs) * w2)
            if P.size > 2:
                P[1:-1] *= 2.0
            elif P.size == 2:
                P[1] *= 2.0
            seg_powers.append(_band_power(freqs_seg, P, band_lo, band_hi))

        band_powers.append(np.array(seg_powers, dtype=np.float64))

    if not band_powers or band_powers[0].size == 0:
        n = len(chs)
        return np.eye(n), ch_labels

    mat = np.vstack(band_powers)  # (n_ch, n_segments)
    R = _corr_matrix(mat)
    return R, ch_labels


# ---------------------------------------------------------------------------
# Heatmap widget (pyqtgraph-based)
# ---------------------------------------------------------------------------

class _HeatmapWidget(pg.GraphicsLayoutWidget):
    """Wraps a pyqtgraph ImageItem + axis labels into a reusable widget."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setBackground("w")

        # Custom axes with tick labels
        self._x_axis = pg.AxisItem(orientation="bottom", pen="k", textPen="k")
        self._y_axis = pg.AxisItem(orientation="left", pen="k", textPen="k")

        self._plot = self.addPlot(
            axisItems={"bottom": self._x_axis, "left": self._y_axis},
        )
        self._plot.setAspectLocked(True)
        self._plot.invertY(True)  # row 0 at top

        # Hide default mouse-drag rectangle
        self._plot.setMenuEnabled(False)

        # Image item
        lut = self._make_rdbu_lut()
        self._img = pg.ImageItem(axisOrder="row-major")
        self._img.setLookupTable(lut)
        self._img.setLevels([-1.0, 1.0])
        self._plot.addItem(self._img)

        # Colour bar (manual via a second plot)
        self._cb_plot = self.addPlot(row=0, col=1)
        self._cb_plot.setMaximumWidth(80)
        self._cb_plot.setMinimumWidth(60)
        self._cb_bar = pg.ColorBarItem(
            values=(-1, 1),
            colorMap=self._make_rdbu_cmap(),
            label="Pearson r",
            interactive=False,
            orientation="v",
        )
        self._cb_bar.setImageItem(self._img, insert_in=self._cb_plot)

    # ------------------------------------------------------------------
    def set_data(self, R: np.ndarray, labels: List[str]) -> None:
        n = R.shape[0]
        if n == 0:
            self._img.clear()
            return

        # Always use RdBu over [-1, 1]. The colour bar was initialised with
        # values=(-1, 1) and never needs to change — the ImageItem levels
        # drive the actual colour mapping.
        self._img.setLevels([-1.0, 1.0])

        self._img.setImage(R, autoLevels=False)

        ticks_x = [(i + 0.5, lab) for i, lab in enumerate(labels)]
        ticks_y = [(i + 0.5, lab) for i, lab in enumerate(labels)]
        self._x_axis.setTicks([ticks_x])
        self._y_axis.setTicks([ticks_y])

        # Adjust font size based on channel count
        font_size = max(5, min(10, int(130 / max(n, 1))))
        style = {"font-size": f"{font_size}pt"}
        self._x_axis.setStyle(tickFont=QtGui.QFont("monospace", font_size))
        self._y_axis.setStyle(tickFont=QtGui.QFont("monospace", font_size))

        self._plot.setXRange(0, n, padding=0.01)
        self._plot.setYRange(0, n, padding=0.01)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_rdbu_lut() -> np.ndarray:
        """Red-White-Blue LUT (256 steps): –1→red, 0→white, +1→blue."""
        n = 256
        lut = np.zeros((n, 3), dtype=np.uint8)
        half = n // 2
        for i in range(half):
            t = i / (half - 1)
            lut[i] = [255, int(255 * t), int(255 * t)]
        for i in range(half, n):
            t = (i - half) / (n - half - 1) if n - half - 1 > 0 else 1.0
            lut[i] = [int(255 * (1 - t)), int(255 * (1 - t)), 255]
        return lut

    @staticmethod
    def _make_rdbu_cmap() -> pg.ColorMap:
        """Red-White-Blue colour map for the colour bar."""
        pos = np.array([0.0, 0.5, 1.0])
        color = np.array([
            [255,   0,   0, 255],
            [255, 255, 255, 255],
            [  0,   0, 255, 255],
        ], dtype=np.uint8)
        return pg.ColorMap(pos, color)




# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

class PCorrView(ViewBase):
    view_name = "PCorr"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)

        ctrl.addWidget(QtWidgets.QLabel("Band:"))
        self.combo_band = QtWidgets.QComboBox()
        for name, lo, hi in BANDS:
            self.combo_band.addItem(f"{name}  ({lo}–{hi} Hz)", userData=(lo, hi))
        ctrl.addWidget(self.combo_band)

        self.chk_visible = QtWidgets.QCheckBox("Use visible window")
        self.chk_visible.setChecked(True)
        self.chk_visible.setToolTip(
            "Compute band power from the time window currently visible in the time-series view."
        )
        ctrl.addWidget(self.chk_visible)

        self.chk_eeg_only = QtWidgets.QCheckBox("EEG channels only")
        self.chk_eeg_only.setChecked(True)
        self.chk_eeg_only.setToolTip(
            "Exclude aux channels (VEOG, HEOG, ECG, …) from the matrix."
        )
        ctrl.addWidget(self.chk_eeg_only)

        self.chk_no_artifact = QtWidgets.QCheckBox("Artifact=0 only")
        self.chk_no_artifact.setChecked(False)
        self.chk_no_artifact.setToolTip(
            "Exclude samples where the artifact channel is non-zero before computing correlations."
        )
        ctrl.addWidget(self.chk_no_artifact)

        ctrl.addStretch(1)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color: #586e75;")  # muted
        ctrl.addWidget(self.lbl_info)

        # Heatmap
        self._heatmap = _HeatmapWidget()
        layout.addWidget(self._heatmap, 1)

        # Internal state
        self.data: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.fs: int = 1
        self._i0: int = 0
        self._i1: Optional[int] = None
        self._artifact_model: Optional[ArtifactModel] = None

        # Wiring
        self.combo_band.currentIndexChanged.connect(lambda _: self._refresh())
        self.chk_visible.toggled.connect(lambda _: self._refresh())
        self.chk_eeg_only.toggled.connect(lambda _: self._refresh())
        self.chk_no_artifact.toggled.connect(lambda _: self._refresh())

    # ------------------------------------------------------------------
    def set_recording(
        self,
        eeg_path: Optional[Path],
        eeg: Dict[str, Any],
        data: np.ndarray,
        labels: List[str],
        fs: int,
        artifact_model: ArtifactModel,
    ) -> None:
        self.data = data
        self.labels = labels
        self.fs = fs
        self._i0 = 0
        self._i1 = data.shape[1] if data is not None else None
        self._artifact_model = artifact_model
        self._refresh()

    def set_visible_window_samples(self, i0: int, i1: int) -> None:
        self._i0 = int(i0)
        self._i1 = int(i1)
        if self.chk_visible.isChecked():
            self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        if self.data is None or self.data.size == 0:
            self._heatmap.set_data(np.empty((0, 0)), [])
            return

        n_ch, n_samp = self.data.shape

        i0, i1 = 0, n_samp
        if self.chk_visible.isChecked() and self._i1 is not None:
            i0 = max(0, min(n_samp - 2, self._i0))
            i1 = max(i0 + 2, min(n_samp, self._i1))

        band_data = self.combo_band.currentData()
        if band_data is None:
            return
        flo, fhi = band_data

        # Build artifact mask for the window
        clean_mask: Optional[np.ndarray] = None
        if (
            self.chk_no_artifact.isChecked()
            and self._artifact_model is not None
            and self._artifact_model.artifact_mask is not None
        ):
            seg_mask = self._artifact_model.artifact_mask[i0:i1]  # True where artifact != 0
            clean_mask = ~seg_mask  # True where clean

        R, ch_labels = _compute_band_corr(
            self.data,
            self.labels,
            self.fs,
            i0,
            i1,
            flo,
            fhi,
            eeg_only=self.chk_eeg_only.isChecked(),
            clean_mask=clean_mask,
        )

        self._heatmap.set_data(R, ch_labels)
        dur_s = (i1 - i0) / self.fs
        self.lbl_info.setText(
            f"{len(ch_labels)} channels · {dur_s:.1f} s window"
        )
