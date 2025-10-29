"""Chronological slicing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..utils.io import write_lines


@dataclass
class SliceConfig:
    total_slices: int = 8
    members: int = 1000
    non_members: int = 1000
    past_members: int = 500
    future_non_members: int = 500


def assign_temporal_slices(df: pd.DataFrame, *, total_slices: int = 8) -> pd.DataFrame:
    """Assign a `slice_id` column based on chronological order."""
    if "discharge_time" not in df.columns:
        raise ValueError("DataFrame must include `discharge_time`.")

    df = df.sort_values("discharge_time").copy()
    quantiles = np.linspace(0, 1, total_slices + 1)
    bins = df["discharge_time"].quantile(quantiles).to_numpy()
    # Ensure strictly increasing bins to avoid duplicates when data is synthetic.
    bins[-1] += pd.Timedelta(seconds=1)
    df["slice_id"] = pd.cut(df["discharge_time"], bins=bins, labels=False, include_lowest=True)
    return df


def enforce_token_budget(df: pd.DataFrame, tokens_per_slice: int) -> pd.DataFrame:
    """Trim datasets so that each slice stays under the defined token budget."""
    df = df.sort_values(["slice_id", "discharge_time"]).copy()
    trimmed_frames = []
    for slice_id, group in df.groupby("slice_id"):
        cumulative = group["tokens_estimate"].cumsum()
        mask = cumulative <= tokens_per_slice
        trimmed_frames.append(group.loc[mask])
    return pd.concat(trimmed_frames, ignore_index=True)


def build_member_panels(df: pd.DataFrame, *, config: SliceConfig, artifact_dir: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Select member/non-member panels per slice and persist ID lists."""
    rng = np.random.default_rng(0)
    panels: Dict[int, Dict[str, np.ndarray]] = {}

    global_holdout = df[df["split_tag"] == "global_holdout"] if "global_holdout" in df["split_tag"].unique() else None

    for slice_id, group in df.groupby("slice_id"):
        slice_members = group[group["split_tag"] == "train"]["subject_id"].unique()
        if len(slice_members) == 0:
            continue

        member_count = min(config.members, len(slice_members))
        members = rng.choice(slice_members, size=member_count, replace=False)

        if global_holdout is not None and not global_holdout.empty:
            pool = global_holdout["subject_id"].unique()
        else:
            pool = df[df["slice_id"] != slice_id]["subject_id"].unique()
        if len(pool) == 0:
            continue
        non_member_count = min(config.non_members, len(pool))
        non_members = rng.choice(pool, size=non_member_count, replace=False)

        past_count = min(config.past_members, len(slice_members))
        past_members = rng.choice(slice_members, size=past_count, replace=False)
        future_pool = df[df["slice_id"] > slice_id]["subject_id"].unique()
        future_count = min(config.future_non_members, len(future_pool)) if len(future_pool) else 0
        future_non_members = (
            rng.choice(future_pool, size=future_count, replace=False) if future_count else np.array([], dtype=int)
        )

        panels[slice_id] = {
            "members": members,
            "non_members": non_members,
            "past_members": past_members,
            "future_non_members": future_non_members,
        }

        for key, values in panels[slice_id].items():
            write_lines(Path(artifact_dir) / f"slice_{slice_id}" / "ids" / f"{key}.txt", map(str, values))

    return panels
