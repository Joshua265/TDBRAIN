"""
EEG .npy viewer (EO/EC) with participant scanning + TSV metadata.

Features
- Recursively scans a root folder for .npy EEG files and groups them by (sub-xxxx, ses-x).
- EO / EC selector per participant (+ session).
- Fast, smooth time scrolling using PyQtGraph with clip-to-view + downsampling.
- Stacked-channel view with adjustable spacing/gain.
- Artifact overlays for any eeg["artifacts"][key] arrays shaped (N, 2) interpreted as [start_sample, end_sample].
- Optional participant metadata TSV viewer (participants_ID column like "sub-19681349").

Tested on the attached example files (dict-based np.load output with keys like Fs, data, labels, artifacts, info).

Install
  pip install pyqtgraph PySide6 numpy pandas

Run
  python eeg_viewer.py --root /path/to/dataset --participants-tsv /path/to/participants.tsv

Or open a single file:
  python eeg_viewer.py --file /path/to/sub-...restEC...npy
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg

# --- Qt binding (prefer PySide6) ---
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception:  # pragma: no cover
    # Fallbacks (best-effort)
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore


def _qt_horizontal():
    # PySide6 / PyQt6 moved enums under Qt.Orientation
    return getattr(
        getattr(QtCore.Qt, "Orientation", QtCore.Qt), "Horizontal", QtCore.Qt.Horizontal
    )


# --- Heuristics for extracting IDs/conditions from filenames ---
SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
SES_RE = re.compile(r"(ses-[A-Za-z0-9]+)")
COND_RE = re.compile(r"rest(EC|EO)\b")


@dataclass(frozen=True)
class RecordingKey:
    sub: str
    ses: str

    def label(self) -> str:
        return f"{self.sub} / {self.ses}"


def load_eeg_dict(path: Path) -> Dict:
    """Loads the .npy and returns a dict (handles both direct dict and 0d object array)."""
    obj = np.load(str(path), allow_pickle=True)
    if isinstance(obj, dict):
        return obj
    # If saved as 0d object array containing a dict
    if isinstance(obj, np.ndarray) and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    raise TypeError(f"Unsupported npy content type: {type(obj)} in {path}")


def extract_segments(artifacts: Dict) -> Dict[str, np.ndarray]:
    """
    Returns only artifact entries that look like (N,2) segments in sample indices.
    Many other keys can contain sample-wise masks (e.g., (channels, samples)) or lists.
    """
    segs: Dict[str, np.ndarray] = {}
    for k, v in artifacts.items():
        if (
            isinstance(v, np.ndarray)
            and v.ndim == 2
            and v.shape[1] == 2
            and v.shape[0] > 0
        ):
            segs[k] = v.astype(np.int64, copy=False)
    return segs


def scan_recordings(root: Path) -> Dict[RecordingKey, Dict[str, Path]]:
    """
    Recursively finds .npy EEG files and groups them by (sub, ses) and condition (EC/EO).
    Only considers files whose name contains 'sub-' and 'restEC'/'restEO'.
    """
    out: Dict[RecordingKey, Dict[str, Path]] = {}
    for p in root.rglob("*.npy"):
        name = p.name
        msub = SUB_RE.search(name)
        mcond = COND_RE.search(name)
        if not (msub and mcond):
            continue
        sub = msub.group(1)
        mses = SES_RE.search(str(p))
        ses = mses.group(1) if mses else "ses-?"
        cond = mcond.group(1)  # 'EC' or 'EO'
        key = RecordingKey(sub=sub, ses=ses)
        out.setdefault(key, {})[cond] = p
    return out


def safe_json(obj) -> str:
    """Pretty JSON for dict-like objects, with ndarray summaries."""

    def default(o):
        if isinstance(o, np.ndarray):
            return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    return json.dumps(obj, indent=2, sort_keys=True, default=default)


class EEGViewer(QtWidgets.QMainWindow):
    def __init__(
        self,
        recordings: Dict[RecordingKey, Dict[str, Path]],
        participants_tsv: Optional[Path] = None,
        initial_file: Optional[Path] = None,
    ):
        super().__init__()
        self.setWindowTitle("EEG .npy Viewer (EO/EC)")
        self.resize(1350, 850)

        # Performance-related defaults
        pg.setConfigOptions(antialias=False)  # faster
        # You can try OpenGL acceleration if it helps on your machine:
        # pg.setConfigOptions(useOpenGL=True)

        self.recordings = recordings
        self.participants_df = None  # type: Optional[pd.DataFrame]
        if participants_tsv and participants_tsv.exists():
            self.participants_df = pd.read_csv(participants_tsv, sep="\t", dtype=str)

        # Current recording state
        self.eeg_path: Optional[Path] = None
        self.eeg: Optional[Dict] = None
        self.data: Optional[np.ndarray] = None  # (n_ch, n_samp) float32
        self.time: Optional[np.ndarray] = None  # (n_samp,) float32
        self.fs: int = 1
        self.labels: List[str] = []
        self.segments: Dict[str, np.ndarray] = {}
        self.region_items: Dict[str, List[pg.LinearRegionItem]] = {}

        # UI
        self._build_ui()

        # Populate and load initial
        self._populate_participants()
        if initial_file:
            self.load_file(initial_file)
        else:
            # pick first available EC/EO file if present
            if self.recordings:
                first_key = sorted(self.recordings.keys(), key=lambda k: k.label())[0]
                conds = self.recordings[first_key]
                preferred = conds.get("EC") or conds.get("EO")
                if preferred:
                    self._set_selector(first_key, "EC" if "EC" in conds else "EO")
                    self.load_file(preferred)

    # ---------- UI construction ----------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left control panel
        left = QtWidgets.QFrame()
        left.setMinimumWidth(340)
        left.setMaximumWidth(460)
        left_layout = QtWidgets.QVBoxLayout(left)
        layout.addWidget(left)

        # Participant/session selector
        grp_sel = QtWidgets.QGroupBox("Selection")
        form = QtWidgets.QFormLayout(grp_sel)

        self.combo_subject = QtWidgets.QComboBox()
        self.combo_condition = QtWidgets.QComboBox()
        self.combo_condition.addItems(["EC", "EO"])

        self.btn_open_file = QtWidgets.QPushButton("Open .npy file…")
        self.btn_rescan = QtWidgets.QPushButton("Rescan root…")

        form.addRow("Participant / Session:", self.combo_subject)
        form.addRow("Condition:", self.combo_condition)
        form.addRow(self.btn_open_file)
        form.addRow(self.btn_rescan)
        left_layout.addWidget(grp_sel)

        # Time navigation
        grp_nav = QtWidgets.QGroupBox("Time Navigation")
        nav = QtWidgets.QGridLayout(grp_nav)

        self.spin_window = QtWidgets.QDoubleSpinBox()
        self.spin_window.setRange(1.0, 600.0)
        self.spin_window.setDecimals(2)
        self.spin_window.setSingleStep(1.0)
        self.spin_window.setValue(30.0)

        self.spin_start = QtWidgets.QDoubleSpinBox()
        self.spin_start.setRange(0.0, 1e9)
        self.spin_start.setDecimals(3)
        self.spin_start.setSingleStep(0.25)

        self.slider = QtWidgets.QSlider(_qt_horizontal())
        self.slider.setRange(0, 0)

        nav.addWidget(QtWidgets.QLabel("Window (s):"), 0, 0)
        nav.addWidget(self.spin_window, 0, 1)
        nav.addWidget(QtWidgets.QLabel("Start (s):"), 1, 0)
        nav.addWidget(self.spin_start, 1, 1)
        nav.addWidget(self.slider, 2, 0, 1, 2)

        left_layout.addWidget(grp_nav)

        # Display settings
        grp_disp = QtWidgets.QGroupBox("Display")
        disp = QtWidgets.QGridLayout(grp_disp)

        self.spin_gain = QtWidgets.QDoubleSpinBox()
        self.spin_gain.setRange(0.01, 100.0)
        self.spin_gain.setDecimals(3)
        self.spin_gain.setValue(1.0)
        self.spin_gain.setSingleStep(0.1)

        self.spin_spacing = QtWidgets.QDoubleSpinBox()
        self.spin_spacing.setRange(1.0, 1000.0)
        self.spin_spacing.setDecimals(1)
        self.spin_spacing.setValue(120.0)
        self.spin_spacing.setSingleStep(10.0)

        disp.addWidget(QtWidgets.QLabel("Gain:"), 0, 0)
        disp.addWidget(self.spin_gain, 0, 1)
        disp.addWidget(QtWidgets.QLabel("Spacing (µV):"), 1, 0)
        disp.addWidget(self.spin_spacing, 1, 1)

        left_layout.addWidget(grp_disp)

        # Artifacts checkboxes area
        grp_art = QtWidgets.QGroupBox("Artifact Overlays")
        self.art_layout = QtWidgets.QVBoxLayout(grp_art)
        self.art_layout.addWidget(
            QtWidgets.QLabel("Load a file to see available artifacts.")
        )
        left_layout.addWidget(grp_art)

        # Metadata + file info tabs
        tabs = QtWidgets.QTabWidget()

        self.txt_participant = QtWidgets.QPlainTextEdit()
        self.txt_participant.setReadOnly(True)
        tabs.addTab(self.txt_participant, "Participant TSV")

        self.txt_fileinfo = QtWidgets.QPlainTextEdit()
        self.txt_fileinfo.setReadOnly(True)
        tabs.addTab(self.txt_fileinfo, "File metadata")

        self.txt_artifacts = QtWidgets.QPlainTextEdit()
        self.txt_artifacts.setReadOnly(True)
        tabs.addTab(self.txt_artifacts, "Artifacts")

        left_layout.addWidget(tabs, 1)

        # Right plot area
        right = QtWidgets.QFrame()
        right_layout = QtWidgets.QVBoxLayout(right)
        layout.addWidget(right, 1)

        self.glw = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.glw, 1)

        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=False, alpha=0.25)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getViewBox().setDefaultPadding(0.0)

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Lines
        self.curves: List[pg.PlotDataItem] = []

        # Coalesced updates for smooth slider dragging
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_plot)

        # Wiring
        self.combo_subject.currentIndexChanged.connect(self._on_subject_changed)
        self.combo_condition.currentTextChanged.connect(self._on_condition_changed)
        self.btn_open_file.clicked.connect(self._open_file_dialog)
        self.btn_rescan.clicked.connect(self._rescan_dialog)

        self.spin_window.valueChanged.connect(self._on_window_changed)
        self.spin_start.valueChanged.connect(self._on_start_spin_changed)
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.spin_gain.valueChanged.connect(self._schedule_update)
        self.spin_spacing.valueChanged.connect(self._schedule_update)

    def _populate_participants(self):
        self.combo_subject.blockSignals(True)
        self.combo_subject.clear()
        keys = sorted(self.recordings.keys(), key=lambda k: k.label())
        for k in keys:
            self.combo_subject.addItem(k.label(), userData=k)
        self.combo_subject.blockSignals(False)

    def _set_selector(self, key: RecordingKey, cond: str):
        # best-effort set current subject and condition without extra updates
        idx = self.combo_subject.findText(key.label())
        if idx >= 0:
            self.combo_subject.setCurrentIndex(idx)
        idxc = self.combo_condition.findText(cond)
        if idxc >= 0:
            self.combo_condition.setCurrentIndex(idxc)

    # ---------- event handlers ----------
    def _open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open EEG .npy", "", "NumPy files (*.npy)"
        )
        if path:
            self.load_file(Path(path))

    def _rescan_dialog(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset root to scan"
        )
        if not root:
            return
        self.recordings = scan_recordings(Path(root))
        self._populate_participants()
        self.status.showMessage(
            f"Scanned {root}: {len(self.recordings)} participant/session entries", 5000
        )

    def _on_subject_changed(self, _):
        key = self.combo_subject.currentData()
        if not isinstance(key, RecordingKey):
            return
        cond = self.combo_condition.currentText().strip()
        self._try_load_selection(key, cond)

    def _on_condition_changed(self, cond: str):
        key = self.combo_subject.currentData()
        if not isinstance(key, RecordingKey):
            return
        self._try_load_selection(key, cond.strip())

    def _try_load_selection(self, key: RecordingKey, cond: str):
        conds = self.recordings.get(key, {})
        p = conds.get(cond)
        if p:
            self.load_file(p)
        else:
            # If requested condition not present, fall back to whichever exists
            fallback = conds.get("EC") or conds.get("EO")
            if fallback:
                self.load_file(fallback)

    def _on_slider_changed(self, val: int):
        if self.time is None:
            return
        start_s = val / 1000.0
        self.spin_start.blockSignals(True)
        self.spin_start.setValue(start_s)
        self.spin_start.blockSignals(False)
        self._schedule_update()

    def _on_start_spin_changed(self, start_s: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(start_s * 1000.0)))
        self.slider.blockSignals(False)
        self._schedule_update()

    def _on_window_changed(self, _win_s: float):
        """Keep start controls consistent when the window length changes."""
        if self.data is None:
            self._schedule_update()
            return
        n_samp = self.data.shape[1]
        duration = float(n_samp) / float(self.fs)
        win_s = float(self.spin_window.value())
        max_start = max(0.0, duration - win_s)

        # Update ranges
        self.spin_start.setRange(0.0, max_start)
        cur = float(self.spin_start.value())
        if cur > max_start:
            self.spin_start.setValue(max_start)

        self.slider.setRange(0, int(round(max_start * 1000.0)))
        self.slider.setValue(int(round(float(self.spin_start.value()) * 1000.0)))

        self._schedule_update()

    def _schedule_update(self):
        # Coalesce rapid UI changes into ~60 FPS updates
        self._update_timer.start(16)

    # ---------- loading + plotting ----------
    def load_file(self, path: Path):
        try:
            eeg = load_eeg_dict(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        self.eeg_path = path
        self.eeg = eeg
        self.fs = int(eeg.get("Fs", 1))

        data = np.squeeze(np.asarray(eeg.get("data")))
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            # handle (1, ch, samples)
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Data error", f"Unexpected data shape: {data.shape}"
                )
                return

        # store as float32 for faster GUI updates
        self.data = data.astype(np.float32, copy=False)
        n_ch, n_samp = self.data.shape
        self.time = np.arange(n_samp, dtype=np.float32) / float(self.fs)

        labels = eeg.get("labels")
        if isinstance(labels, np.ndarray):
            self.labels = [str(x) for x in labels.tolist()]
        elif isinstance(labels, list):
            self.labels = [str(x) for x in labels]
        else:
            self.labels = [f"ch{i}" for i in range(n_ch)]

        # artifacts
        artifacts = (
            eeg.get("artifacts", {})
            if isinstance(eeg.get("artifacts", {}), dict)
            else {}
        )
        self.segments = extract_segments(artifacts)

        # (re)build plot items
        self._rebuild_curves()
        self._rebuild_artifacts_overlays()

        # update controls ranges
        duration = float(n_samp) / float(self.fs)
        self.spin_start.setRange(0.0, max(0.0, duration - self.spin_window.value()))
        self.spin_start.setValue(0.0)

        # slider in milliseconds for smoothness but still integer
        self.slider.setRange(
            0, int(max(0.0, duration - self.spin_window.value()) * 1000.0)
        )
        self.slider.setValue(0)

        self._update_metadata_panels()
        self.update_plot()

    def _rebuild_curves(self):
        self.plot.clear()
        self.curves = []
        if self.data is None:
            return

        n_ch, _ = self.data.shape
        # one curve per channel
        for _ in range(n_ch):
            c = pg.PlotDataItem()
            # Performance options: clip + downsampling
            c.setClipToView(True)
            c.setDownsampling(auto=True, method="peak")  # good for EEG-like signals
            self.plot.addItem(c)
            self.curves.append(c)

    def _rebuild_artifacts_overlays(self):
        # Remove old checkboxes
        while self.art_layout.count():
            item = self.art_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.region_items = {}

        if not self.segments:
            self.art_layout.addWidget(
                QtWidgets.QLabel("No segment-like artifacts found (N×2 arrays).")
            )
            return

        # Palette similar to your matplotlib example
        colors = {
            "VEOG": (255, 0, 0, 60),
            "HEOG": (255, 165, 0, 60),
            "EMGtrl": (0, 0, 255, 50),
            "JUMPtrl": (0, 255, 255, 40),
            "KURTtrl": (0, 255, 0, 40),
            "SWINGtrl": (255, 105, 180, 45),
            "EBtrl": (128, 0, 128, 45),
        }

        lbl = QtWidgets.QLabel("Show/hide segment overlays:")
        lbl.setWordWrap(True)
        self.art_layout.addWidget(lbl)

        for k in sorted(self.segments.keys()):
            cb = QtWidgets.QCheckBox(k)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, kk=k: self._toggle_artifact(kk, checked))
            self.art_layout.addWidget(cb)

            rgba = colors.get(k, (180, 180, 180, 35))
            brush = pg.mkBrush(*rgba)
            pen = pg.mkPen(None)

            items: List[pg.LinearRegionItem] = []
            for seg in self.segments[k]:
                s0, s1 = int(seg[0]), int(seg[1])
                if s1 <= s0:
                    continue
                r = pg.LinearRegionItem(
                    values=(s0 / self.fs, s1 / self.fs),
                    orientation="vertical",
                    brush=brush,
                    pen=pen,
                    movable=False,
                )
                r.setZValue(-10)  # behind curves
                self.plot.addItem(r)
                items.append(r)
            self.region_items[k] = items

        self.art_layout.addStretch(1)

    def _toggle_artifact(self, key: str, visible: bool):
        for item in self.region_items.get(key, []):
            item.setVisible(visible)

    def _update_metadata_panels(self):
        if self.eeg is None or self.eeg_path is None:
            self.txt_fileinfo.setPlainText("")
            self.txt_artifacts.setPlainText("")
            return

        # File metadata (show all keys but avoid dumping full data arrays)
        meta = {k: v for k, v in self.eeg.items() if k != "data"}
        meta["__file__"] = str(self.eeg_path)
        meta["__data__"] = {
            "shape": list(np.squeeze(np.asarray(self.eeg["data"])).shape),
            "dtype": str(np.asarray(self.eeg["data"]).dtype),
        }
        self.txt_fileinfo.setPlainText(safe_json(meta))

        # Artifacts summary
        artifacts = self.eeg.get("artifacts", {})
        if isinstance(artifacts, dict):
            summary = {}
            for k, v in artifacts.items():
                if isinstance(v, np.ndarray):
                    summary[k] = {
                        "type": "ndarray",
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                    }
                else:
                    try:
                        summary[k] = {"type": type(v).__name__, "len": len(v)}  # type: ignore
                    except Exception:
                        summary[k] = {"type": type(v).__name__}
            self.txt_artifacts.setPlainText(safe_json(summary))
        else:
            self.txt_artifacts.setPlainText(str(artifacts))

        # Participant TSV (match by participants_ID like "sub-19681349")
        self._update_participant_tsv()

    def _update_participant_tsv(self):
        if self.participants_df is None or self.eeg_path is None:
            self.txt_participant.setPlainText("(No participants.tsv loaded)")
            return

        msub = SUB_RE.search(self.eeg_path.name)
        if not msub:
            self.txt_participant.setPlainText(
                "(Could not parse subject ID from filename)"
            )
            return
        sub = msub.group(1)

        df = self.participants_df
        if "participants_ID" not in df.columns:
            self.txt_participant.setPlainText(
                "(participants.tsv missing 'participants_ID' column)"
            )
            return

        matches = df[df["participants_ID"].astype(str) == sub]
        if matches.empty:
            self.txt_participant.setPlainText(f"(No TSV row found for {sub})")
            return

        # If multiple rows per participant, show them all compactly
        rows = matches.to_dict(orient="records")
        self.txt_participant.setPlainText(safe_json(rows if len(rows) > 1 else rows[0]))

    def update_plot(self):
        if self.data is None or self.time is None or not self.curves:
            return

        n_ch, n_samp = self.data.shape
        win_s = float(self.spin_window.value())
        start_s = float(self.spin_start.value())
        start_s = max(0.0, min(start_s, (n_samp / self.fs) - win_s))
        end_s = min(start_s + win_s, n_samp / self.fs)

        i0 = int(round(start_s * self.fs))
        i1 = int(round(end_s * self.fs))
        i0 = max(0, min(i0, n_samp - 1))
        i1 = max(i0 + 2, min(i1, n_samp))

        x = self.time[i0:i1]
        gain = float(self.spin_gain.value())
        spacing = float(self.spin_spacing.value())

        # Offsets: top channel at y=0, then negative downwards
        offsets = -np.arange(n_ch, dtype=np.float32) * spacing

        # Update y-axis labels/ticks
        ticks = [
            (float(offsets[i]), self.labels[i] if i < len(self.labels) else f"ch{i}")
            for i in range(n_ch)
        ]
        self.plot.getAxis("left").setTicks([ticks])

        # Update each curve
        for ch in range(n_ch):
            y = self.data[ch, i0:i1] * gain + offsets[ch]
            self.curves[ch].setData(x, y)

        # View ranges
        self.plot.setXRange(float(x[0]), float(x[-1]), padding=0.0)
        ymin = float(offsets[-1] - spacing * 1.0)
        ymax = float(offsets[0] + spacing * 1.0)
        self.plot.setYRange(ymin, ymax, padding=0.0)

        # Status line
        self.status.showMessage(
            f"{self.eeg_path.name if self.eeg_path else ''}   Fs={self.fs} Hz   "
            f"t=[{start_s:.3f}, {end_s:.3f}] s   samples=[{i0}, {i1})   ch={n_ch}",
            0,
        )


def main():
    ap = argparse.ArgumentParser()
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
        "--file",
        type=str,
        default=None,
        help="Open a specific .npy file (overrides selection)",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else None
    tsv = (
        Path(args.participants_tsv).expanduser().resolve()
        if args.participants_tsv
        else None
    )
    initial_file = Path(args.file).expanduser().resolve() if args.file else None

    if root and root.exists():
        recs = scan_recordings(root)
    else:
        # Fall back to current directory
        recs = scan_recordings(Path.cwd())

    app = QtWidgets.QApplication([])
    win = EEGViewer(recs, participants_tsv=tsv, initial_file=initial_file)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
