"""Likelihood ratio attack utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LiRAResult:
    llr: np.ndarray
    normalized_llr: np.ndarray


def compute_llr(log_probs_ft: np.ndarray, log_probs_base: np.ndarray) -> LiRAResult:
    """Compute per-example log-likelihood ratios."""
    if log_probs_ft.shape != log_probs_base.shape:
        raise ValueError("Fine-tuned and base log probs must share shape")
    llr = (log_probs_ft - log_probs_base).sum(axis=1)
    denom = np.maximum(np.abs(log_probs_base).sum(axis=1), 1e-6)
    normalized = llr / denom
    return LiRAResult(llr=llr, normalized_llr=normalized)


def temporal_llr(llr_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Summaries for temporal leakage analysis."""
    mean_traj = llr_matrix.mean(axis=0)
    std_traj = llr_matrix.std(axis=0)
    return {"mean": mean_traj, "std": std_traj}
