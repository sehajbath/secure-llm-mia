"""Evaluation metrics for membership inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def auc_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    pr, rc, _ = precision_recall_curve(labels, scores)
    return {
        "auc": float(roc_auc),
        "roc_curve": (fpr, tpr),
        "pr_curve": (rc, pr),
    }


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    if len(fpr) == 0:
        return 0.0
    idx = np.searchsorted(fpr, target_fpr, side="left")
    idx = min(idx, len(tpr) - 1)
    return float(tpr[idx])


@dataclass
class CalibrationResult:
    ece: float
    bin_counts: np.ndarray


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, *, bins: int = 15) -> CalibrationResult:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(probs, bin_edges, right=True)
    ece = 0.0
    counts = np.zeros(bins)
    for b in range(1, bins + 1):
        mask = indices == b
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        counts[b - 1] = mask.sum()
        ece += np.abs(bin_acc - bin_conf) * (mask.sum() / len(labels))
    return CalibrationResult(ece=float(ece), bin_counts=counts)


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    if len(fpr) == 0:
        return 0.0, 0.0
    idx = np.searchsorted(fpr, target_fpr, side="left")
    idx = min(idx, len(fpr) - 1)
    return thresholds[idx], tpr[idx]
