"""Feature stacking and ensemble attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

try:  # XGBoost is optional.
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    normalized = (features - mean) / std
    return normalized, mean, std


@dataclass
class EnsembleModels:
    logistic: LogisticRegression
    xgboost: object | None
    mean: np.ndarray
    std: np.ndarray


def train_ensemble(features: np.ndarray, labels: np.ndarray) -> EnsembleModels:
    normalized, mean, std = standardize_features(features)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(normalized, labels)

    xgb_model = None
    if XGBClassifier is not None:
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
        xgb_model.fit(normalized, labels)

    return EnsembleModels(logistic=logreg, xgboost=xgb_model, mean=mean, std=std)


def predict_proba(models: EnsembleModels, features: np.ndarray) -> Dict[str, np.ndarray]:
    normalized = (features - models.mean) / models.std
    out = {
        "logistic": models.logistic.predict_proba(normalized)[:, 1],
    }
    if models.xgboost is not None:
        out["xgboost"] = models.xgboost.predict_proba(normalized)[:, 1]
    return out
