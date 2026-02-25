from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _as_2d_segments(v: Any) -> Optional[np.ndarray]:
    if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2 and v.shape[0] > 0:
        return v.astype(np.int64, copy=False)
    return None


def mask_to_segments(mask_1d: np.ndarray) -> np.ndarray:
    """Convert boolean/int mask (n_samp,) into Nx2 segments [start,end) in sample indices."""
    if mask_1d.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    m = mask_1d.astype(np.int8) != 0
    d = np.diff(m.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    n = min(starts.size, ends.size)
    segs = np.column_stack([starts[:n], ends[:n]]).astype(np.int64)
    segs = segs[segs[:, 1] > segs[:, 0]]
    return segs


def intersect_segments(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersect two Nx2 segment lists (both [start,end) sample indices)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    out: List[Tuple[int, int]] = []
    ia = 0
    ib = 0
    a = a[np.argsort(a[:, 0])]
    b = b[np.argsort(b[:, 0])]
    while ia < a.shape[0] and ib < b.shape[0]:
        s = max(int(a[ia, 0]), int(b[ib, 0]))
        e = min(int(a[ia, 1]), int(b[ib, 1]))
        if e > s:
            out.append((s, e))
        if a[ia, 1] < b[ib, 1]:
            ia += 1
        else:
            ib += 1
    if not out:
        return np.zeros((0, 2), dtype=np.int64)

    arr = np.array(out, dtype=np.int64)
    arr = arr[np.argsort(arr[:, 0])]
    merged: List[List[int]] = [[int(arr[0, 0]), int(arr[0, 1])]]
    for s, e in arr[1:]:
        s = int(s)
        e = int(e)
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.array(merged, dtype=np.int64)


class ArtifactModel:
    """
    Holds artifact annotations and provides masked detector segments + inspection helpers.

    Masking rule:
    - If a boolean artifact channel (label == 'artifacts') exists, overlays are masked:
      detector_segments_masked = detector_segments ∩ artifact_mask_segments.
    - If no artifact channel exists, overlays show raw detector segments.
    """

    def __init__(
        self, eeg: Dict[str, Any], data: np.ndarray, labels: List[str], fs: int
    ):
        self.eeg = eeg
        self.data = data
        self.labels = labels
        self.fs = fs

        self._artifact_dict: Dict[str, Any] = (
            eeg.get("artifacts", {})
            if isinstance(eeg.get("artifacts", {}), dict)
            else {}
        )

        self.detector_segments: Dict[str, np.ndarray] = {}
        for k, v in self._artifact_dict.items():
            seg = _as_2d_segments(v)
            if seg is not None:
                self.detector_segments[k] = seg

        # sample-wise (channels x samples) arrays for inspection, e.g. EMGsamps
        self.sample_masks: Dict[str, np.ndarray] = {}
        for k, v in self._artifact_dict.items():
            if (
                isinstance(v, np.ndarray)
                and v.ndim == 2
                and v.shape[-1] == data.shape[1]
                and str(k).lower().endswith("samps")
            ):
                self.sample_masks[k] = v

        self.artifact_mask_channel_index = self._find_artifact_channel_index(labels)
        if self.artifact_mask_channel_index is not None:
            ch = data[self.artifact_mask_channel_index]
            self.artifact_mask = ch != 0
            self.artifact_segments = mask_to_segments(self.artifact_mask)
        else:
            self.artifact_mask = None
            self.artifact_segments = None

        self.detector_segments_masked: Dict[str, np.ndarray] = {}
        if self.artifact_segments is not None:
            for k, seg in self.detector_segments.items():
                self.detector_segments_masked[k] = intersect_segments(
                    seg, self.artifact_segments
                )
        else:
            self.detector_segments_masked = dict(self.detector_segments)

    @staticmethod
    def _find_artifact_channel_index(labels: List[str]) -> Optional[int]:
        for i, lab in enumerate(labels):
            if str(lab).strip().lower() in {"artifact", "artifacts", "artfct", "art"}:
                return i
        return None

    def has_global_mask(self) -> bool:
        return self.artifact_segments is not None

    def global_segments(self) -> np.ndarray:
        return (
            self.artifact_segments
            if self.artifact_segments is not None
            else np.zeros((0, 2), dtype=np.int64)
        )

    def segments_for(self, key: str, masked: bool = True) -> np.ndarray:
        if masked:
            return self.detector_segments_masked.get(
                key, np.zeros((0, 2), dtype=np.int64)
            )
        return self.detector_segments.get(key, np.zeros((0, 2), dtype=np.int64))

    def artifact_coverage_percent(self) -> Optional[float]:
        if self.artifact_mask is None:
            return None
        return 100.0 * float(np.mean(self.artifact_mask.astype(np.float32)))

    def describe_segment(self, key: str, seg: Tuple[int, int]) -> Dict[str, Any]:
        s0, s1 = int(seg[0]), int(seg[1])
        t0, t1 = s0 / float(self.fs), s1 / float(self.fs)
        duration = t1 - t0

        overlaps: List[str] = []
        for other_k, other_segs in self.detector_segments_masked.items():
            if other_k == key or other_segs.size == 0:
                continue
            if np.any((other_segs[:, 0] < s1) & (other_segs[:, 1] > s0)):
                overlaps.append(other_k)

        contrib = []
        for mk, arr in self.sample_masks.items():
            win = arr[:, s0:s1]
            if win.size == 0:
                continue
            nonzero = np.where(np.max(np.abs(win), axis=1) > 0)[0]
            if nonzero.size:
                mx = np.max(np.abs(win), axis=1)
                top = nonzero[np.argsort(mx[nonzero])[::-1]][:5]
                top_names = [
                    self.labels[i] if i < len(self.labels) else f"ch{i}" for i in top
                ]
                contrib.append(
                    {
                        "mask": mk,
                        "n_channels": int(nonzero.size),
                        "top_channels": top_names,
                    }
                )

        mask_cov = None
        if self.artifact_mask is not None:
            m = self.artifact_mask[s0:s1]
            mask_cov = 100.0 * float(np.mean(m.astype(np.float32))) if m.size else None

        return {
            "key": key,
            "start_sample": s0,
            "end_sample": s1,
            "start_s": t0,
            "end_s": t1,
            "duration_s": duration,
            "masked_by_artifact_channel": self.has_global_mask(),
            "artifact_channel_coverage_percent_in_window": mask_cov,
            "cooccurring_detectors": overlaps,
            "contributors": contrib,
        }
