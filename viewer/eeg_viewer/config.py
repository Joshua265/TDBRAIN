from __future__ import annotations

import re
from typing import Dict, Tuple

# --- Filename parsing ---
SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
SES_RE = re.compile(r"(ses-[A-Za-z0-9]+)")

# FIX: allow underscore/dash/dot/end after condition (restEC_eeg..., restEO-..., restEC.npy, etc.)
# Old pattern used \b which fails on underscores (underscore is a "word char").
COND_RE = re.compile(r"rest(?:[_-]?)?(EC|EO)(?=$|[_\-.])", re.IGNORECASE)

# --- Artifact tooltips / UI text ---
ARTIFACT_TOOLTIPS: Dict[str, str] = {
    "ARTIFACT_MASK": "Final boolean artifact mask channel (label 'artifacts') converted to segments.",
    "EMGtrl": "EMG (muscle) bursts (often high-frequency).",
    "JUMPtrl": "Step/jump artifacts (baseline shifts / electrode pops).",
    "KURTtrl": "High kurtosis (spiky bumps / outliers).",
    "SWINGtrl": "Extreme voltage swing (large peak-to-peak).",
    "EBtrl": "Residual eye blinks after EOG regression.",
    "VEOG": "Vertical EOG-related activity windows (blinks).",
    "HEOG": "Horizontal EOG-related activity windows (saccades).",
}

# Colors RGBA for overlays
ARTIFACT_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "ARTIFACT_MASK": (160, 160, 160, 45),
    "VEOG": (255, 0, 0, 60),
    "HEOG": (255, 165, 0, 60),
    "EMGtrl": (0, 0, 255, 50),
    "JUMPtrl": (0, 255, 255, 40),
    "KURTtrl": (0, 255, 0, 40),
    "SWINGtrl": (255, 105, 180, 45),
    "EBtrl": (128, 0, 128, 45),
}

ARTIFACT_GLOSSARY_TEXT = (
    "Artifact glossary (practical)\n\n"
    "Representation\n"
    "- Detectors often store Nx2 windows as [start_sample, end_sample).\n"
    "- Many files also include a boolean time-series channel named 'artifacts' marking samples considered artifact.\n\n"
    "Common detectors\n"
    "- EMGtrl: muscle bursts (often high-frequency).\n"
    "- JUMPtrl: step/jump artifacts (baseline shifts).\n"
    "- KURTtrl: high kurtosis (spiky bumps/outliers).\n"
    "- SWINGtrl: extreme peak-to-peak voltage swing.\n"
    "- EBtrl: residual eye blinks after EOG regression.\n"
    "- VEOG/HEOG: eye activity windows from vertical/horizontal EOG.\n\n"
    "Viewer behavior\n"
    "- Overlays are intersected with the boolean 'artifacts' channel when present.\n"
    "- Click a shaded region to open Artifact Inspector (timing + overlaps + contributing masks).\n"
)

CHANNEL_GLOSSARY_TEXT = (
    "Channel glossary (TDBRAIN-style files)\n\n"
    "Auxiliary channels often included alongside EEG:\n"
    "- VEOG/HEOG: vertical/horizontal EOG-derived channels capturing eye blinks and saccades.\n"
    "- OrbOcc: orbicularis oculi (around-the-eye muscle) channel.\n"
    "- MASS: masseter (jaw) muscle EMG.\n"
    "- Erbs: ECG measured at the clavicle/Erb's point area.\n\n"
    "Tip: For EEG-only analytics (PSD/topography), exclude these aux channels.\n"
)
