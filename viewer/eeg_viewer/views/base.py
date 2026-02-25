from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..qt_compat import QtWidgets
from ..artifacts import ArtifactModel


class ViewBase(QtWidgets.QWidget):
    view_name: str = "Base"

    def set_recording(
        self,
        eeg_path: Optional[Path],
        eeg: Dict[str, Any],
        data: "Any",
        labels: List[str],
        fs: int,
        artifact_model: ArtifactModel,
    ) -> None:
        raise NotImplementedError
