"""Deterministic seeding helpers."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:  # torch is optional on CPU-only environments.
    import torch
except Exception:  # pragma: no cover - torch not installed in lightweight envs.
    torch = None  # type: ignore


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed python, numpy, and torch RNGs.

    Parameters
    ----------
    seed: int
        Seed value to broadcast.
    deterministic_torch: bool, default True
        Enable deterministic CUDA kernels if available.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def seed_worker(worker_id: int) -> None:
    """Pytorch DataLoader worker seeding hook."""
    seed = np.random.get_state()[1][0] + worker_id
    set_global_seed(int(seed % (2**32)))


def generate_seeds(base_seed: int, count: int) -> list[int]:
    """Utility to expand a base seed into a deterministic list of seeds."""
    rng = np.random.default_rng(base_seed)
    return rng.integers(low=0, high=2**31 - 1, size=count).tolist()
