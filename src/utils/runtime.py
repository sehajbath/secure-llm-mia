"""Runtime configuration helpers (subset vs full dataset toggles)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RunModeConfig:
    """Lightweight configuration describing how much data to process."""

    name: str
    max_rows: Optional[int]
    description: str


RUN_MODE_REGISTRY: dict[str, RunModeConfig] = {
    "subset": RunModeConfig(
        name="subset",
        max_rows=2000,
        description="Quick debugging subset (<=2k rows) for lightweight Colab smoke tests.",
    ),
    "full": RunModeConfig(
        name="full",
        max_rows=None,
        description="Process the entire dataset (requires longer runtimes and more storage).",
    ),
}


def current_run_mode(name: Optional[str] = None) -> RunModeConfig:
    """Resolve the active run mode from argument or environment variable."""

    requested = (name or os.getenv("SECURE_LLM_MIA_RUN_MODE", "subset")).lower()
    if requested not in RUN_MODE_REGISTRY:
        requested = "subset"
    return RUN_MODE_REGISTRY[requested]


def limit_dataframe(df, run_mode: RunModeConfig):
    """Return a dataframe limited by the run mode's max rows (if defined)."""

    if run_mode.max_rows is None:
        return df
    return df.head(run_mode.max_rows)
