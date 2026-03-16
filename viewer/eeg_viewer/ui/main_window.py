from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..qt_compat import QtCore, QtGui, QtWidgets
from ..utils import safe_json
from ..io import RecordingKey, load_eeg_dict, scan_recordings
from ..artifacts import ArtifactModel
from ..config import (
    ARTIFACT_GLOSSARY_TEXT,
    CHANNEL_GLOSSARY_TEXT,
)
from ..views.timeseries import TimeSeriesView
from ..views.psd import PSDView
from ..views.pcorr import PCorrView
from ..views.passage_hist import PassageHistView
from ..views.base import ViewBase


class PlaceholderView(ViewBase):
    view_name = "(Add view…)"

    def __init__(self, text: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        layout.addWidget(label, 1)

    def set_recording(self, *args, **kwargs) -> None:
        return


class EEGViewer(QtWidgets.QMainWindow):
    def __init__(
        self,
        recordings: Dict[RecordingKey, Dict[str, Path]],
        participants_tsv: Optional[Path] = None,
        initial_file: Optional[Path] = None,
    ):
        super().__init__()
        self.setWindowTitle("EEG .npy Viewer (EO/EC) — modular")
        self.resize(1450, 900)

        self.recordings = recordings
        self.participants_df: Optional[pd.DataFrame] = None
        if participants_tsv and participants_tsv.exists():
            self.participants_df = pd.read_csv(participants_tsv, sep="\t", dtype=str)

        self.eeg_path: Optional[Path] = None
        self.eeg: Optional[Dict[str, Any]] = None
        self.data: Optional[np.ndarray] = None
        self.fs: int = 1
        self.labels: List[str] = []
        self.art_model: Optional[ArtifactModel] = None

        self._build_ui()
        self._build_menu()
        self._populate_participants()

        if initial_file:
            self.load_file(initial_file)
        elif self.recordings:
            first_key = sorted(self.recordings.keys(), key=lambda k: k.label())[0]
            conds = self.recordings[first_key]
            preferred = conds.get("EC") or conds.get("EO")
            if preferred:
                self._set_selector(first_key, "EC" if "EC" in conds else "EO")
                self.load_file(preferred)

    # ---------- menu ----------
    def _build_menu(self):
        menubar = self.menuBar()
        helpm = menubar.addMenu("Help")

        act_gloss = QtGui.QAction("Artifact glossary…", self)
        act_gloss.triggered.connect(
            lambda: self._show_text_dialog("Artifact glossary", ARTIFACT_GLOSSARY_TEXT)
        )
        helpm.addAction(act_gloss)

        act_chan = QtGui.QAction("Channel glossary…", self)
        act_chan.triggered.connect(
            lambda: self._show_text_dialog("Channel glossary", CHANNEL_GLOSSARY_TEXT)
        )
        helpm.addAction(act_chan)

        act_about = QtGui.QAction("About", self)
        act_about.triggered.connect(self._show_about)
        helpm.addAction(act_about)

    def _show_text_dialog(self, title: str, text: str):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(720, 560)
        lay = QtWidgets.QVBoxLayout(dlg)
        box = QtWidgets.QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText(text)
        lay.addWidget(box, 1)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()

    def _show_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "EEG .npy viewer for TDBRAIN-style outputs.\n\n"
            "- Time navigation UI removed; use mouse pan/zoom.\n"
            "- Artifact overlays are masked by the boolean 'artifacts' channel when present.\n"
            "- Click overlay regions for an Artifact Inspector.\n"
            "- Modular view framework: add new views to the right-side tabs.",
        )

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left panel
        left = QtWidgets.QFrame()
        left.setMinimumWidth(360)
        left.setMaximumWidth(520)
        left_layout = QtWidgets.QVBoxLayout(left)
        layout.addWidget(left)

        # Selection
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

        # Display
        grp_disp = QtWidgets.QGroupBox("Display (Time series)")
        disp = QtWidgets.QGridLayout(grp_disp)

        self.spin_gain = QtWidgets.QDoubleSpinBox()
        self.spin_gain.setRange(0.01, 100.0)
        self.spin_gain.setDecimals(3)
        self.spin_gain.setValue(1.0)
        self.spin_gain.setSingleStep(0.1)

        self.spin_spacing = QtWidgets.QDoubleSpinBox()
        self.spin_spacing.setRange(1.0, 2000.0)
        self.spin_spacing.setDecimals(1)
        self.spin_spacing.setValue(120.0)
        self.spin_spacing.setSingleStep(10.0)

        disp.addWidget(QtWidgets.QLabel("Gain:"), 0, 0)
        disp.addWidget(self.spin_gain, 0, 1)
        disp.addWidget(QtWidgets.QLabel("Spacing (µV):"), 1, 0)
        disp.addWidget(self.spin_spacing, 1, 1)
        left_layout.addWidget(grp_disp)

        # QC summary
        grp_qc = QtWidgets.QGroupBox("QC summary")
        qcl = QtWidgets.QVBoxLayout(grp_qc)
        self.lbl_qc = QtWidgets.QLabel("Load a file to see QC.")
        self.lbl_qc.setWordWrap(True)
        qcl.addWidget(self.lbl_qc)
        left_layout.addWidget(grp_qc)


        # Metadata tabs
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

        # Right: views
        right = QtWidgets.QFrame()
        right_layout = QtWidgets.QVBoxLayout(right)
        layout.addWidget(right, 1)

        self.view_tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.view_tabs, 1)

        self.time_view = TimeSeriesView()
        self.time_view.sigVisibleRangeChanged.connect(self._on_visible_range_changed)
        self.time_view.sigVisibleRangeChanged.connect(
            lambda _start_s, _end_s, i0, i1: self.psd_view.set_visible_window_samples(
                i0, i1
            )
        )
        self.time_view.sigVisibleRangeChanged.connect(
            lambda _start_s, _end_s, i0, i1: self.pcorr_view.set_visible_window_samples(
                i0, i1
            )
        )
        self.view_tabs.addTab(self.time_view, self.time_view.view_name)

        self.psd_view = PSDView()
        self.view_tabs.addTab(self.psd_view, self.psd_view.view_name)

        self.pcorr_view = PCorrView()
        self.view_tabs.addTab(self.pcorr_view, self.pcorr_view.view_name)

        self.passage_hist_view = PassageHistView()
        self.view_tabs.addTab(self.passage_hist_view, self.passage_hist_view.view_name)

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Wiring
        self.combo_subject.currentIndexChanged.connect(self._on_subject_changed)
        self.combo_condition.currentTextChanged.connect(self._on_condition_changed)
        self.btn_open_file.clicked.connect(self._open_file_dialog)
        self.btn_rescan.clicked.connect(self._rescan_dialog)
        self.spin_gain.valueChanged.connect(self._on_display_changed)
        self.spin_spacing.valueChanged.connect(self._on_display_changed)

    # ---------- selection ----------
    def _populate_participants(self):
        self.combo_subject.blockSignals(True)
        self.combo_subject.clear()
        keys = sorted(self.recordings.keys(), key=lambda k: k.label())
        for k in keys:
            self.combo_subject.addItem(k.label(), userData=k)
        self.combo_subject.blockSignals(False)
        self.passage_hist_view.set_all_paths(self._all_recording_paths())

    def _all_recording_paths(self) -> list:
        """Flat list of every .npy Path known in the current recording dict."""
        paths = []
        for cond_dict in self.recordings.values():
            paths.extend(cond_dict.values())
        return paths

    def _set_selector(self, key: RecordingKey, cond: str):
        idx = self.combo_subject.findText(key.label())
        if idx >= 0:
            self.combo_subject.setCurrentIndex(idx)
        idxc = self.combo_condition.findText(cond)
        if idxc >= 0:
            self.combo_condition.setCurrentIndex(idxc)

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
            fallback = conds.get("EC") or conds.get("EO")
            if fallback:
                self.load_file(fallback)

    # ---------- dialogs ----------
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
        self._populate_participants()   # also calls set_all_paths
        self.status.showMessage(
            f"Scanned {root}: {len(self.recordings)} participant/session entries", 5000
        )

    # ---------- display ----------
    def _on_display_changed(self, _):
        self.time_view.set_display(self.spin_gain.value(), self.spin_spacing.value())

    # ---------- loading ----------
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
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Data error", f"Unexpected data shape: {data.shape}"
                )
                return

        self.data = data.astype(np.float32, copy=False)
        n_ch, _ = self.data.shape

        labels = eeg.get("labels")
        if isinstance(labels, np.ndarray):
            self.labels = [str(x) for x in labels.tolist()]
        elif isinstance(labels, list):
            self.labels = [str(x) for x in labels]
        else:
            self.labels = [f"ch{i}" for i in range(n_ch)]

        self.art_model = ArtifactModel(
            eeg=eeg, data=self.data, labels=self.labels, fs=self.fs
        )

        # push to view
        self.time_view.set_recording(
            self.eeg_path, eeg, self.data, self.labels, self.fs, self.art_model
        )
        self.time_view.set_display(self.spin_gain.value(), self.spin_spacing.value())

        self.psd_view.set_recording(
            self.eeg_path, eeg, self.data, self.labels, self.fs, self.art_model
        )

        self.pcorr_view.set_recording(
            self.eeg_path, eeg, self.data, self.labels, self.fs, self.art_model
        )

        self.passage_hist_view.set_recording(
            self.eeg_path, eeg, self.data, self.labels, self.fs, self.art_model
        )

        self._update_metadata_panels()
        self._update_qc_panel()

    # ---------- metadata ----------
    def _update_metadata_panels(self):
        if self.eeg is None or self.eeg_path is None:
            self.txt_fileinfo.setPlainText("")
            self.txt_artifacts.setPlainText("")
            return

        meta = {k: v for k, v in self.eeg.items() if k != "data"}
        meta["__file__"] = str(self.eeg_path)
        meta["__data__"] = {
            "shape": list(np.squeeze(np.asarray(self.eeg["data"])).shape),
            "dtype": str(np.asarray(self.eeg["data"]).dtype),
        }
        self.txt_fileinfo.setPlainText(safe_json(meta))

        artifacts = self.eeg.get("artifacts", {})
        if isinstance(artifacts, dict):
            summary: Dict[str, Any] = {}
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

        self._update_participant_tsv()

    def _update_participant_tsv(self):
        if self.participants_df is None or self.eeg_path is None:
            self.txt_participant.setPlainText("(No participants.tsv loaded)")
            return

        import re
        from ..config import SUB_RE

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

        rows = matches.to_dict(orient="records")

        def _keep_value(v) -> bool:
            # drop NaN / None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
            return True

        filtered_rows = []
        for r in rows:
            filtered_rows.append({k: v for k, v in r.items() if _keep_value(v)})

        self.txt_participant.setPlainText(
            safe_json(filtered_rows if len(filtered_rows) > 1 else filtered_rows[0])
        )

    # ---------- QC + status ----------
    def _update_qc_panel(self):
        if self.art_model is None:
            self.lbl_qc.setText("(No QC)")
            return
        cov = self.art_model.artifact_coverage_percent()
        if cov is None:
            self.lbl_qc.setText(
                "Artifact channel not found. Showing raw detector windows."
            )
        else:
            self.lbl_qc.setText(
                f"Artifact-channel coverage: {cov:.1f}%\n"
                f"Overlay masking: ON (detectors ∩ artifacts==1)"
            )

    def _on_visible_range_changed(self, start_s: float, end_s: float, i0: int, i1: int):
        if self.eeg_path is None or self.data is None:
            return
        n_ch = self.data.shape[0]
        self.status.showMessage(
            f"{self.eeg_path.name}   Fs={self.fs} Hz   "
            f"t=[{start_s:.3f}, {end_s:.3f}] s   samples=[{i0}, {i1})   ch={n_ch}",
            0,
        )
