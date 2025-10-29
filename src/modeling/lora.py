"""LoRA/QLoRA helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LoRAHyperParams:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


def compute_gradient_accumulation(tokens_per_step: int, micro_batch: int, avg_tokens_per_sample: int) -> int:
    """Derive gradient accumulation steps to meet a global token budget per optimizer step."""
    if avg_tokens_per_sample <= 0:
        raise ValueError("avg_tokens_per_sample must be positive")
    samples_per_step = tokens_per_step / avg_tokens_per_sample
    grad_accum = max(1, int(round(samples_per_step / micro_batch)))
    return grad_accum


def lora_config_dict(hparams: LoRAHyperParams) -> Dict[str, Any]:
    """Return a dictionary to instantiate PEFT LoraConfig."""
    return {
        "r": hparams.r,
        "lora_alpha": hparams.alpha,
        "lora_dropout": hparams.dropout,
        "target_modules": list(hparams.target_modules),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def describe_setup(base_model: str, load_in_4bit: bool) -> str:
    mode = "QLoRA" if load_in_4bit else "LoRA"
    return f"{mode} setup targeting {base_model}."
