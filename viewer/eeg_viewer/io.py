from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import SUB_RE, SES_RE, COND_RE


@dataclass(frozen=True)
class RecordingKey:
    sub: str
    ses: str

    def label(self) -> str:
        return f"{self.sub} / {self.ses}"


def load_eeg_dict(path: Path) -> Dict[str, Any]:
    """Loads the .npy and returns a dict (handles both direct dict and 0d object array)."""
    obj = np.load(str(path), allow_pickle=True)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.ndarray) and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
    raise TypeError(f"Unsupported npy content type: {type(obj)} in {path}")


def parse_sub(name: str) -> Optional[str]:
    m = SUB_RE.search(name)
    return m.group(1) if m else None


def parse_ses(path_str: str) -> str:
    m = SES_RE.search(path_str)
    return m.group(1) if m else "ses-?"


def parse_cond(filename: str) -> Optional[str]:
    m = COND_RE.search(filename)
    if not m:
        return None
    return m.group(1).upper()


def scan_recordings(root: Path) -> Dict[RecordingKey, Dict[str, Path]]:
    """
    Recursively finds .npy EEG files and groups them by (sub, ses) and condition (EC/EO).
    Condition matches restEC_... / restEO_... (underscore-safe).
    """
    out: Dict[RecordingKey, Dict[str, Path]] = {}
    for p in root.rglob("*.npy"):
        sub = parse_sub(p.name)
        cond = parse_cond(p.name)
        if not (sub and cond):
            continue
        ses = parse_ses(str(p))
        key = RecordingKey(sub=sub, ses=ses)
        out.setdefault(key, {})[cond] = p
    return out
