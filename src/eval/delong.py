"""DeLong test implementation adapted for educational use."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = x.shape[0]
    sort_idx = np.argsort(x)
    sorted_x = x[sort_idx]
    midranks = np.zeros(J, dtype=float)

    i = 0
    while i < J:
        j = i
        while j < J and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1)
        midranks[i:j] = midrank
        i = j
    out = np.empty(J, dtype=float)
    out[sort_idx] = midranks + 1
    return out


def _delong_covariance(midranks: np.ndarray, label_positives: int, label_negatives: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = midranks[:label_positives]
    neg = midranks[label_positives:]

    V10 = (pos - pos.mean()) / label_negatives
    V01 = (neg - neg.mean()) / label_positives
    return V10, V01


def auc_covariance(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]

    label_positives = int(labels.sum())
    label_negatives = len(labels) - label_positives

    midranks = _compute_midrank(scores)
    auc_value = (midranks[:label_positives].sum() - label_positives * (label_positives + 1) / 2) / (
        label_positives * label_negatives
    )

    V10, V01 = _delong_covariance(midranks, label_positives, label_negatives)
    s10 = np.cov(V10, bias=True)
    s01 = np.cov(V01, bias=True)
    return float(auc_value), float(s10 / label_positives + s01 / label_negatives)


def delong_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    """Return z-statistic and p-value comparing two ROC AUCs."""
    auc_a, var_a = auc_covariance(labels, scores_a)
    auc_b, var_b = auc_covariance(labels, scores_b)

    covariance = var_a + var_b
    if covariance == 0:
        return 0.0, 1.0
    z_score = (auc_a - auc_b) / math.sqrt(covariance)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))
    return float(z_score), float(p_value)
