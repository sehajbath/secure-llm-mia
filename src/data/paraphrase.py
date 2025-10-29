"""Paraphrase utilities for stability-based attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from ..utils.seed import set_global_seed


@dataclass
class ParaphraseConfig:
    seed: int = 17
    similarity_threshold: float = 0.82
    length_ratio_bounds: Tuple[float, float] = (0.75, 1.25)
    variants_per_example: Tuple[int, int] = (2, 4)


def _simple_synonym_swap(text: str) -> str:
    """Naive synonym swap using an illustrative dictionary."""
    synonyms = {
        "patient": "individual",
        "treatment": "therapy",
        "diagnosis": "assessment",
        "hospital": "medical center",
        "doctor": "clinician",
    }
    words = text.split()
    swapped = [synonyms.get(word.lower(), word) for word in words]
    return " ".join(swapped)


def _reverse_sentence(text: str) -> str:
    return " ".join(text.split()[::-1])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embed(text: str) -> np.ndarray:
    """Toy embedding via hashing for lightweight testing."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=768)


def generate_paraphrases(texts: Iterable[str], config: ParaphraseConfig) -> List[List[str]]:
    """Generate synthetic paraphrases meeting similarity constraints."""
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    paraphrases: List[List[str]] = []
    for text in texts:
        base_emb = _embed(text)
        num_variants = rng.integers(config.variants_per_example[0], config.variants_per_example[1] + 1)
        variants: List[str] = []
        for _ in range(num_variants):
            candidate = _simple_synonym_swap(text)
            if rng.random() < 0.3:
                candidate = _reverse_sentence(candidate)
            cand_emb = _embed(candidate)
            sim = cosine_similarity(base_emb, cand_emb)
            length_ratio = len(candidate) / max(len(text), 1)
            if sim < config.similarity_threshold or not (config.length_ratio_bounds[0] <= length_ratio <= config.length_ratio_bounds[1]):
                continue
            variants.append(candidate)
        paraphrases.append(variants)
    return paraphrases
