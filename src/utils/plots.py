"""Matplotlib plotting helpers to keep notebooks tidy."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def save_roc_curve(fpr: Sequence[float], tpr: Sequence[float], *, title: str, path: Path) -> None:
    """Save a ROC curve figure."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_bar_plot(labels: Sequence[str], values: Sequence[float], *, title: str, ylabel: str, path: Path) -> None:
    """Simple helper for bar plots with tight layout."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#2a9d8f")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
