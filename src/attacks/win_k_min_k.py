"""Win-k and Min-k% attack utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def win_k_fraction(win_matrix: np.ndarray) -> np.ndarray:
    """Compute fraction of tokens that fall within top-k."""
    return win_matrix.mean(axis=1)


def min_k_percent_loss(nll: np.ndarray, percent: float) -> np.ndarray:
    """Average loss over the worst-k% tokens."""
    if not 0 < percent <= 1:
        raise ValueError("percent must be in (0, 1]")
    sorted_loss = np.sort(nll, axis=1)[:, ::-1]
    k = max(1, int(round(percent * nll.shape[1])))
    return sorted_loss[:, :k].mean(axis=1)


def aggregate_features(win_dict: Dict[str, np.ndarray], nll: np.ndarray, worst_percents: Iterable[float]) -> Dict[str, np.ndarray]:
    """Combine Win-k fractions with worst-percentile losses."""
    features = {}
    for key, values in win_dict.items():
        label = key.replace("win@", "")
        features[f"win_frac_{label}"] = win_k_fraction(values.astype(float))
    for percent in worst_percents:
        features[f"min_loss_top_{int(percent*100)}pct"] = min_k_percent_loss(nll, percent)
    return features
