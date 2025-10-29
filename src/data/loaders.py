"""Dataset loading stubs with synthetic fallbacks for development."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..utils.seed import set_global_seed


@dataclass
class LoaderConfig:
    """Configuration for loading raw MIMIC-style data."""
    root: Path
    table: str
    seed: int = 17
    limit: Optional[int] = 1000


def _synthetic_notes(limit: int) -> pd.DataFrame:
    """Create a toy note dataset for quick notebook smoke tests."""
    rng = np.random.default_rng(0)
    subject_ids = np.arange(limit)
    records = []
    for subject_id in subject_ids:
        discharge_offset = rng.integers(0, 365)
        records.append(
            {
                "subject_id": int(subject_id),
                "hadm_id": int(subject_id * 10),
                "discharge_time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(discharge_offset)),
                "note_text": f"Synthetic discharge summary for patient {subject_id}.",
            }
        )
    return pd.DataFrame(records)


def load_notes(config: LoaderConfig) -> pd.DataFrame:
    """Load note-style data.

    Returns a synthetic DataFrame unless a real path is provided. Replace with secure
    data access logic (e.g., PhysioNet downloads) before production use.
    """
    set_global_seed(config.seed)
    if not config.root.exists():
        return _synthetic_notes(config.limit or 1000)

    # TODO: Implement credentialed reads from PhysioNet storage.
    raise NotImplementedError("Secure data loading is not implemented in this template.")


def estimate_token_counts(df: pd.DataFrame, chars_per_token: float = 4.0) -> pd.DataFrame:
    """Attach a naive token estimate column for planning budgets."""
    df = df.copy()
    text_lengths = df["note_text"].str.len()
    df["tokens_estimate"] = (text_lengths / chars_per_token).round().astype(int)
    return df


def tag_split(df: pd.DataFrame, *, train_frac: float = 0.8) -> pd.DataFrame:
    """Assign train/val/test splits by patient ID with disjointness enforced."""
    df = df.sort_values("discharge_time").copy()
    unique_subjects = df["subject_id"].unique()
    n_train = int(len(unique_subjects) * train_frac)
    n_val = max(1, int(len(unique_subjects) * 0.1))

    split_map = {}
    split_map.update({sid: "train" for sid in unique_subjects[:n_train]})
    split_map.update({sid: "val" for sid in unique_subjects[n_train : n_train + n_val]})
    split_map.update({sid: "test" for sid in unique_subjects[n_train + n_val :]})

    df["split_tag"] = df["subject_id"].map(split_map)
    return df


def export_canonical(df: pd.DataFrame, path: Path) -> Path:
    """Export canonical parquet used by later notebooks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_canonical(path: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load canonical parquet if it exists, else return empty DF."""
    if not path.exists():
        return pd.DataFrame(columns=list(columns or []))
    return pd.read_parquet(path, columns=list(columns) if columns else None)
