"""Bootstrap utilities."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def bootstrap_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int = 2000,
    seed: int = 17,
    confidence: float = 0.95,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(labels)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        values.append(metric_fn(labels[idx], scores[idx]))
    values_arr = np.array(values)
    lower = np.percentile(values_arr, (1 - confidence) / 2 * 100)
    upper = np.percentile(values_arr, (1 + confidence) / 2 * 100)
    return {
        "estimate": float(metric_fn(labels, scores)),
        "ci_low": float(lower),
        "ci_high": float(upper),
    }
