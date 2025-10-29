"""Loss/confidence based membership inference scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LossConfidenceScores:
    mean_nll: float
    entropy: float
    max_prob: float


def summarize_token_stats(nll: np.ndarray, entropy: np.ndarray, max_prob: np.ndarray) -> LossConfidenceScores:
    """Aggregate token-level stats into example-level scores."""
    return LossConfidenceScores(
        mean_nll=float(nll.mean()),
        entropy=float(entropy.mean()),
        max_prob=float(max_prob.mean()),
    )


def score_examples(nll: np.ndarray, entropy: np.ndarray, max_prob: np.ndarray) -> Dict[str, np.ndarray]:
    """Return per-example aggregate scores."""
    return {
        "mean_nll": nll.mean(axis=1),
        "entropy": entropy.mean(axis=1),
        "max_prob": max_prob.mean(axis=1),
        "perplexity": np.exp(nll.mean(axis=1)),
    }
