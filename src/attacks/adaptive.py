"""Adaptive/stability-based attack stubs."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def stability_probe(log_probs: Sequence[np.ndarray]) -> Dict[str, float]:
    """Measure variance across repeated queries."""
    stacked = np.stack(log_probs, axis=0)
    variance = stacked.var(axis=0).mean()
    max_variation = np.abs(stacked.max(axis=0) - stacked.min(axis=0)).mean()
    return {
        "stability_var": float(variance),
        "stability_range": float(max_variation),
    }
