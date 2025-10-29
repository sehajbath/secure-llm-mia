"""Project-wide constants for the secure-llm-mia repo."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(os.environ.get("SECURE_LLM_MIA_ROOT", Path.cwd()))
DATA_CACHE_DIR: Path = PROJECT_ROOT / "data_cache"
ARTIFACT_ROOT: Path = PROJECT_ROOT / "artifacts"


def ensure_directories() -> None:
    """Create core directories lazily when running inside Colab."""
    for path in (DATA_CACHE_DIR, ARTIFACT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
