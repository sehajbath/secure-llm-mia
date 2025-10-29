"""Training utilities with explicit token accounting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple


def _ensure_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass
class TokenBudgetTracker:
    tokens_per_slice: int
    consumed_tokens: int = 0

    def update(self, tokens_in_batch: int) -> bool:
        """Update token usage. Returns True if the budget is exhausted."""
        _ensure_positive(tokens_in_batch, "tokens_in_batch")
        self.consumed_tokens += tokens_in_batch
        return self.consumed_tokens >= self.tokens_per_slice

    @property
    def remaining(self) -> int:
        return max(0, self.tokens_per_slice - self.consumed_tokens)


def simulate_training_loop(batches: Iterable[Tuple[int, int]], tracker: TokenBudgetTracker) -> int:
    """Simulate a training loop and return the number of processed batches.

    Parameters
    ----------
    batches: Iterable[Tuple[int, int]]
        Each tuple holds `(micro_batch_size, avg_tokens_per_sample)`.
    tracker: TokenBudgetTracker
        Tracks the running token consumption.
    """
    processed = 0
    for micro_batch, avg_tokens in batches:
        batch_tokens = micro_batch * avg_tokens
        exhausted = tracker.update(batch_tokens)
        processed += 1
        if exhausted:
            break
    return processed
