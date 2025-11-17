"""Loader utilities for the MIMIC-IV-Ext-BHC dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.runtime import RunModeConfig

EXPECTED_COLUMNS = {"note_id", "input", "target", "input_tokens", "target_tokens"}


@dataclass
class BHCDataConfig:
    csv_path: Path
    run_mode: RunModeConfig
    subset_rows: Optional[int] = None
    seed: int = 17


def load_bhc_dataframe(config: BHCDataConfig) -> pd.DataFrame:
    """Load the BHC CSV with optional row limiting based on run mode."""

    if not config.csv_path.exists():
        raise FileNotFoundError(
            f"BHC dataset not found at {config.csv_path}. Upload it to Drive before running this notebook."
        )

    row_limit = config.subset_rows or config.run_mode.max_rows
    read_kwargs = {"dtype": {"note_id": str}}
    if row_limit:
        read_kwargs["nrows"] = row_limit

    df = pd.read_csv(config.csv_path, **read_kwargs)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"BHC CSV is missing required columns: {sorted(missing)}")
    return df


def bhc_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BHC dataframe into the canonical schema used downstream."""

    canonical = df.copy()
    canonical["subject_id"] = canonical["note_id"].astype(str)
    canonical["hadm_id"] = canonical["subject_id"]

    # Synthesize a monotonic timestamp for chronological slicing.
    canonical = canonical.sort_values("note_id").reset_index(drop=True)
    baseline = pd.Timestamp("2018-01-01")
    canonical["discharge_time"] = baseline + pd.to_timedelta(canonical.index, unit="h")

    instruction = "Summarize the discharge note into a Brief Hospital Course."

    def render_example(row: pd.Series) -> str:
        return (
            f"{instruction}\n\n"
            f"### Discharge Note\n{row['input']}\n\n"
            f"### Expected Brief Hospital Course\n{row['target']}"
        )

    canonical["text"] = canonical.apply(render_example, axis=1)
    canonical["tokens_estimate"] = canonical["input_tokens"] + canonical["target_tokens"]
    canonical = canonical[
        [
            "subject_id",
            "hadm_id",
            "discharge_time",
            "text",
            "tokens_estimate",
            "note_id",
            "input",
            "target",
            "input_tokens",
            "target_tokens",
        ]
    ]
    return canonical
