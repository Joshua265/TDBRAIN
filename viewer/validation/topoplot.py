"""
Topoplot helper — lightweight wrapper around MNE for topographic maps.

Uses mne.viz.plot_topomap with standard 10-20 montage positions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# Standard 10-20 channel list used by the TDBRAIN pipeline (26 EEG channels)
STANDARD_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]


def get_mne_info(
    channels: Optional[List[str]] = None,
    sfreq: float = 500.0,
) -> "mne.Info":
    """
    Create an MNE Info object with standard 10-20 positions.
    """
    if not HAS_MNE:
        raise RuntimeError("MNE is required for topographic maps")

    ch_names = channels or STANDARD_CHANNELS
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="warn")
    return info


def plot_topomap(
    values: Dict[str, float],
    ax: "plt.Axes",
    channels: Optional[List[str]] = None,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_names: bool = False,
) -> Optional[object]:
    """
    Plot a topographic map of scalar values per channel.

    Parameters
    ----------
    values : dict mapping channel name → scalar value
    ax : matplotlib Axes
    channels : channel list to use (defaults to STANDARD_CHANNELS)

    Returns the image object or None if MNE not available.
    """
    if not HAS_MNE:
        ax.text(0.5, 0.5, "MNE not installed\n(topomap unavailable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        return None

    ch_names = channels or STANDARD_CHANNELS
    # Filter to channels that exist in values
    available = [ch for ch in ch_names if ch in values]
    if not available:
        ax.text(0.5, 0.5, "No matching channels",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        return None

    data = np.array([values[ch] for ch in available])
    info = get_mne_info(available)

    im, _ = mne.viz.plot_topomap(
        data, info, axes=ax,
        cmap=cmap, vlim=(vmin, vmax),
        show=False,
        names=available if show_names else None,
        sensors=True,
        contours=6,
    )

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    return im
