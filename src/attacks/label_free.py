"""Reference-free membership inference utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CalibrationPrompt:
    template: str = "Given the context, is the narrative clinically plausible?"

    def format(self, context: str) -> str:
        return f"{self.template}\n\nContext:\n{context}"  # TODO: replace with richer prompts.


def stability_score(log_probs: np.ndarray) -> np.ndarray:
    """Compute a stability score based on prediction entropy."""
    entropy = -np.sum(np.exp(log_probs) * log_probs, axis=-1)
    return 1.0 - entropy / np.log(log_probs.shape[-1])


def summarize_outputs(prompts: List[str], responses: List[str]) -> List[float]:
    """Pseudo scoring using response length as a proxy."""
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must align")
    return [min(1.0, len(resp) / (len(prompt) + 1)) for prompt, resp in zip(prompts, responses)]
