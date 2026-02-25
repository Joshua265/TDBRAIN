from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from ..artifacts import ArtifactModel
from ..qt_compat import QtCore, QtWidgets, Signal
from ..config import ARTIFACT_COLORS, ARTIFACT_TOOLTIPS


class ArtifactRegionItem(pg.LinearRegionItem):
    sigClicked = Signal(object)  # emits self

    def __init__(self, *args, meta: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = meta or {}

    def mouseClickEvent(self, ev):  # type: ignore
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            self.sigClicked.emit(self)
        else:
            ev.ignore()


class TimeSeriesView(QtWidgets.QWidget):
    """
    Time-series view with no external time-navigation UI:
    - Pan/zoom with mouse (PyQtGraph ViewBox).
    - Curves are updated based on the current visible X-range (fast slicing).
    - Artifact overlay controls are integrated at the top of the view.
    """

    view_name = "Time series"
    sigVisibleRangeChanged = Signal(float, float, int, int)  # start_s, end_s, i0, i1

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=False)

        layout = QtWidgets.QVBoxLayout(self)

        # --- Artifact overlay controls (top row, like PSD settings) ---
        self._art_ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(self._art_ctrl)

        self._art_ctrl.addWidget(QtWidgets.QLabel("Artifacts:"))
        self._art_cb_container = QtWidgets.QHBoxLayout()
        self._art_ctrl.addLayout(self._art_cb_container)
        self._art_ctrl.addStretch(1)

        self._art_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}

        # --- Plot ---
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw, 1)

        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=False, alpha=0.25)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getViewBox().setDefaultPadding(0.0)

        self.eeg_path: Optional[Path] = None
        self.data: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.fs: int = 1
        self.time: Optional[np.ndarray] = None

        self.curves: List[pg.PlotDataItem] = []

        self.artifact_model: Optional[ArtifactModel] = None
        self.region_items: Dict[str, List[ArtifactRegionItem]] = {}

        self.gain: float = 1.0
        self.spacing: float = 120.0

        # coalesce view-range changes
        self._range_timer = QtCore.QTimer(self)
        self._range_timer.setSingleShot(True)
        self._range_timer.timeout.connect(self._update_visible_from_view)

        self.plot.getViewBox().sigXRangeChanged.connect(
            lambda *_: self._schedule_range_update()
        )

    def set_display(self, gain: float, spacing: float) -> None:
        self.gain = float(gain)
        self.spacing = float(spacing)
        self._schedule_range_update()

    @staticmethod
    def _is_artifact_channel(label: str) -> bool:
        s = label.strip().lower()
        return s in {"artifact", "artifacts", "artfct", "art"}

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
        self.artifact_model = artifact_model
        self.fs = fs

        # Filter out the boolean artifact channel from display
        keep = [i for i, lab in enumerate(labels) if not self._is_artifact_channel(lab)]
        self.data = data[keep, :]
        self.labels = [labels[i] for i in keep]

        self.time = np.arange(self.data.shape[1], dtype=np.float32) / float(fs)

        # Compute per-channel scale factors so every channel fits within its lane.
        # Uses robust std (IQR-based) to avoid outlier-driven scaling.
        n_ch = self.data.shape[0]
        self._ch_scales = np.ones(n_ch, dtype=np.float32)
        for i in range(n_ch):
            ch = self.data[i]
            q25, q75 = np.percentile(ch, [25, 75])
            iqr = q75 - q25
            robust_std = iqr / 1.35  # IQR → approx std for normal data
            if robust_std > 1e-9:
                self._ch_scales[i] = 1.0 / robust_std

        # Normalize so the median scale is 1.0 (EEG channels dominate)
        med_scale = float(np.median(self._ch_scales))
        if med_scale > 1e-12:
            self._ch_scales /= med_scale

        self._rebuild_curves()
        self._rebuild_artifact_controls_and_overlays()

        # initial view: first 30s (or full if shorter)
        duration = self.data.shape[1] / float(fs)
        self.plot.setXRange(0.0, min(30.0, duration), padding=0.0)
        self._schedule_range_update()

    # ---- artifact controls & overlays ----

    def _rebuild_artifact_controls_and_overlays(self) -> None:
        """Build checkboxes for each artifact type and create overlay regions."""
        # Clear old checkboxes
        self._art_checkboxes.clear()
        while self._art_cb_container.count():
            item = self._art_cb_container.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.clear_artifact_overlays()

        if self.artifact_model is None:
            return

        # Gather all available artifact keys — treat all equally
        segs_by_key: Dict[str, np.ndarray] = {}

        if self.artifact_model.has_global_mask():
            segs_by_key["ARTIFACT_MASK"] = self.artifact_model.global_segments()

        for k in sorted(self.artifact_model.detector_segments.keys()):
            segs_by_key[k] = self.artifact_model.segments_for(k, masked=False)

        if not segs_by_key:
            return

        # Build overlay regions for all keys
        self._build_artifact_overlays(segs_by_key, ARTIFACT_COLORS)

        # Create a checkbox per artifact key
        for k in segs_by_key:
            n_segs = len(segs_by_key[k])
            cb = QtWidgets.QCheckBox(f"{k} ({n_segs})")
            # Default: ARTIFACT_MASK is checked, all others unchecked
            is_default = (k == "ARTIFACT_MASK")
            cb.setChecked(is_default)
            cb.setToolTip(
                ARTIFACT_TOOLTIPS.get(k, "Detector window list (Nx2 sample indices).")
            )
            cb.toggled.connect(
                lambda checked, kk=k: self._set_artifact_visibility(kk, checked)
            )
            self._art_cb_container.addWidget(cb)
            self._art_checkboxes[k] = cb

            # Apply initial visibility
            self._set_artifact_visibility(k, is_default)

    def _build_artifact_overlays(
        self,
        segments_by_key: Dict[str, np.ndarray],
        colors_by_key: Dict[str, Tuple[int, int, int, int]],
    ) -> None:
        self.clear_artifact_overlays()
        if self.artifact_model is None:
            return

        for k, segs in segments_by_key.items():
            rgba = colors_by_key.get(k, (180, 180, 180, 35))
            brush = pg.mkBrush(*rgba)
            pen = pg.mkPen(None)

            items: List[ArtifactRegionItem] = []
            for row_idx, seg in enumerate(segs):
                s0, s1 = int(seg[0]), int(seg[1])
                if s1 <= s0:
                    continue
                meta = self.artifact_model.describe_segment(k, (s0, s1))
                meta["row_index"] = row_idx
                r = ArtifactRegionItem(
                    values=(s0 / self.fs, s1 / self.fs),
                    orientation="vertical",
                    brush=brush,
                    pen=pen,
                    movable=False,
                    meta=meta,
                )
                r.setZValue(-10)
                r.sigClicked.connect(self._on_artifact_region_clicked)
                self.plot.addItem(r)
                items.append(r)
            self.region_items[k] = items

    def _set_artifact_visibility(self, key: str, visible: bool) -> None:
        for item in self.region_items.get(key, []):
            item.setVisible(bool(visible))

    # kept for backward compat if anything else calls it
    def set_artifact_visibility(self, key: str, visible: bool) -> None:
        self._set_artifact_visibility(key, visible)
        cb = self._art_checkboxes.get(key)
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)

    def clear_artifact_overlays(self):
        for items in self.region_items.values():
            for it in items:
                try:
                    self.plot.removeItem(it)
                except Exception:
                    pass
        self.region_items = {}

    # ---- artifact inspector dialog ----

    def _on_artifact_region_clicked(self, region_item: ArtifactRegionItem) -> None:
        meta = region_item.meta or {}
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Artifact inspector")
        dlg.resize(680, 560)
        lay = QtWidgets.QVBoxLayout(dlg)

        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        lay.addWidget(txt, 1)

        key = meta.get("key", "?")
        lines = [
            f"Detector: {key}",
            f"Samples: [{meta.get('start_sample')}, {meta.get('end_sample')})",
            f"Time: {meta.get('start_s'):.3f} – {meta.get('end_s'):.3f} s",
            f"Duration: {meta.get('duration_s'):.3f} s",
        ]

        # Coverage info — same for all types
        if meta.get("masked_by_artifact_channel"):
            cov = meta.get("artifact_channel_coverage_percent_in_window")
            lines.append(
                f"Artifact-channel coverage in window: {cov:.1f}%"
                if cov is not None
                else "Artifact-channel coverage: (unknown)"
            )
        else:
            lines.append("Masked by artifact channel: no (artifact channel not found)")

        overlaps = meta.get("cooccurring_detectors") or []
        lines.append(
            "Co-occurring detectors: " + (", ".join(overlaps) if overlaps else "(none)")
        )

        contrib = meta.get("contributors") or []
        lines.append("")
        lines.append("Detector sample-mask contributors (heuristic):")
        if contrib:
            for c in contrib:
                lines.append(
                    f"- {c.get('mask')}: channels affected={c.get('n_channels')}  top={c.get('top_channels')}"
                )
        else:
            lines.append("- (none / not available)")

        tip = ARTIFACT_TOOLTIPS.get(key)
        if tip:
            lines.append("")
            lines.append("Meaning:")
            lines.append(tip)

        txt.setPlainText("\n".join(lines))

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()

    # ---- internal ----

    def _schedule_range_update(self):
        # ~60 FPS max
        self._range_timer.start(16)

    def _rebuild_curves(self):
        self.plot.clear()
        self.curves = []
        if self.data is None:
            return
        n_ch, _ = self.data.shape

        # choose a stable color cycle
        hues = max(12, min(36, n_ch))  # enough distinct colors without being too noisy

        for i in range(n_ch):
            c = pg.PlotDataItem()
            c.setClipToView(True)
            c.setDownsampling(auto=True, method="peak")

            # Distinct color per channel
            color = pg.intColor(i, hues=hues, values=1.0, maxValue=255)
            c.setPen(pg.mkPen(color, width=1))

            self.plot.addItem(c)
            self.curves.append(c)

    def _update_visible_from_view(self):
        if self.data is None or self.time is None or not self.curves:
            return

        (x0, x1), _ = self.plot.getViewBox().viewRange()
        x0 = float(max(0.0, x0))
        x1 = float(max(x0 + 1e-6, x1))

        n_ch, n_samp = self.data.shape
        duration = n_samp / float(self.fs)
        x1 = min(x1, duration)

        i0 = int(max(0, min(n_samp - 1, round(x0 * self.fs))))
        i1 = int(max(i0 + 2, min(n_samp, round(x1 * self.fs))))

        x = self.time[i0:i1]
        offsets = -np.arange(n_ch, dtype=np.float32) * float(self.spacing)

        ticks = [
            (float(offsets[i]), self.labels[i] if i < len(self.labels) else f"ch{i}")
            for i in range(n_ch)
        ]
        self.plot.getAxis("left").setTicks([ticks])

        for ch in range(n_ch):
            scale = float(self._ch_scales[ch]) if hasattr(self, '_ch_scales') else 1.0
            y = self.data[ch, i0:i1] * float(self.gain) * scale + offsets[ch]
            self.curves[ch].setData(x, y)

        ymin = float(offsets[-1] - float(self.spacing) * 1.0)
        ymax = float(offsets[0] + float(self.spacing) * 1.0)
        self.plot.setYRange(ymin, ymax, padding=0.0)

        self.sigVisibleRangeChanged.emit(x0, x1, i0, i1)
