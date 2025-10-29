"""IO helpers for Colab and local runs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..constants import ARTIFACT_ROOT, PROJECT_ROOT


def project_path(*parts: str) -> Path:
    """Resolve a path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def artifact_path(*parts: str, create: bool = False) -> Path:
    """Resolve paths under the artifact directory.

    Parameters
    ----------
    create: bool
        When True, ensures the parent directory exists.
    """
    path = ARTIFACT_ROOT.joinpath(*parts)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_lines(path: Path, lines: Iterable[str]) -> None:
    """Write newline-separated lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def read_lines(path: Path) -> list[str]:
    """Safely read a file into a list of stripped lines."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
