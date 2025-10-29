"""Log-probability utilities used for attack features."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    return logits - logsumexp


def token_level_stats(logits: np.ndarray, target_ids: np.ndarray, top_k: Tuple[int, ...] = (1, 5, 10, 20)) -> Dict[str, np.ndarray]:
    """Compute per-token log probability, entropy, and top-k hits."""
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError("logits and target_ids must align on prefix dimensions")

    log_probs = log_softmax(logits)
    batch_indices = np.arange(target_ids.shape[0])[:, None]
    token_indices = np.arange(target_ids.shape[1])[None, :]
    nll = -log_probs[batch_indices, token_indices, target_ids]
    max_prob = np.exp(log_probs.max(axis=-1))
    entropy = -np.sum(np.exp(log_probs) * log_probs, axis=-1)

    win_k = {}
    sorted_indices = np.argsort(-log_probs, axis=-1)
    for k in top_k:
        hits = sorted_indices[:, :, :k] == target_ids[..., None]
        win_k[f"win@{k}"] = hits.any(axis=-1)
    return {
        "nll": nll,
        "entropy": entropy,
        "max_prob": np.exp(log_probs.max(axis=-1)),
        "log_probs": log_probs,
        **win_k,
    }
