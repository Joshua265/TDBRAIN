from __future__ import annotations

import json
from typing import Any

import numpy as np


def safe_json(obj: Any) -> str:
    """Pretty JSON for dict-like objects, with ndarray summaries."""

    def default(o: Any):
        if isinstance(o, np.ndarray):
            return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    return json.dumps(obj, indent=2, sort_keys=True, default=default)
