from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from utils.intervention_hooks import (
    DEFAULT_LAST_K,
    DEFAULT_SCOPE,
    assert_scope,
    make_intervention_hook,
)

IPI_OPTION_LETTERS = ("A", "B", "C", "D", "E")
IPI_OPTION_SCORES: dict[str, int] = {
    "A": -2,
    "B": -1,
    "C": 0,
    "D": 1,
    "E": 2,
}
SCORES_ORDERED = (-2, -1, 0, 1, 2)


def option_letter_variants(letter: str) -> list[str]:
    return [letter, f" {letter}", f"\n{letter}", f"{letter}.", f"{letter})"]


def discover_option_token_ids(tokenizer: Any, prompt_text: str) -> dict[int, list[int]]:
    """Discover single-token IDs for A–E answers in chat-template context."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    option_ids: dict[int, list[int]] = {}

    for letter in IPI_OPTION_LETTERS:
        score = IPI_OPTION_SCORES[letter]
        token_ids: set[int] = set()
        for variant in option_letter_variants(letter):
            continuation_ids = tokenizer.encode(
                prompt_text + variant, add_special_tokens=False
            )
            new_ids = continuation_ids[len(prompt_ids) :]
            if len(new_ids) == 1:
                token_ids.add(new_ids[0])
        if not token_ids:
            raise ValueError(
                f"No single-token verbalizer found for option {letter!r} (score {score}). "
                "Try additional variants or inspect the chat template."
            )
        option_ids[score] = sorted(token_ids)

    return option_ids


def expected_ipi_from_logits(
    logits: torch.Tensor,
    option_token_ids: dict[int, list[int]],
) -> float:
    """Expected IPI in [-2, 2] from next-token logits at the answer position."""
    option_logits: list[torch.Tensor] = []
    for score in SCORES_ORDERED:
        token_ids = option_token_ids[score]
        idx = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
        option_logits.append(torch.logsumexp(logits[idx], dim=0))

    stacked = torch.stack(option_logits)
    probs = F.softmax(stacked, dim=0)
    weights = torch.tensor(SCORES_ORDERED, dtype=probs.dtype, device=probs.device)
    return float(torch.sum(probs * weights).item())


def _layer_neuron_multipliers(
    activation_multipliers: Dict[str, float],
) -> Dict[int, Dict[int, float]]:
    layer_neuron_multipliers: Dict[int, Dict[int, float]] = {}
    for feature_name, multiplier in activation_multipliers.items():
        parts = feature_name.split("-")
        layer_idx = int(parts[0].split("_")[1])
        neuron_idx = int(parts[1].split("_")[1])
        layer_neuron_multipliers.setdefault(layer_idx, {})[neuron_idx] = multiplier
    return layer_neuron_multipliers


def forward_last_token_logits(
    wrapper: Any,
    input_ids: torch.Tensor,
    activation_multipliers: Optional[Dict[str, float]] = None,
    intervention_scope: str = DEFAULT_SCOPE,
    last_k: int = DEFAULT_LAST_K,
) -> torch.Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_device = getattr(wrapper, "input_device", wrapper.device)
    input_ids = input_ids.to(input_device)

    if not activation_multipliers:
        with torch.no_grad():
            logits = wrapper.model(input_ids)
    else:
        assert_scope(intervention_scope)
        layer_neuron_multipliers = _layer_neuron_multipliers(activation_multipliers)
        input_len = int(input_ids.shape[1])
        fwd_hooks = [
            (
                f"blocks.{layer_idx}.hook_resid_pre",
                make_intervention_hook(
                    neuron_mults=neuron_mults,
                    input_len=input_len,
                    scope=intervention_scope,
                    last_k=last_k,
                ),
            )
            for layer_idx, neuron_mults in layer_neuron_multipliers.items()
        ]
        with torch.no_grad():
            logits = wrapper.model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)

    return logits[0, -1, :]


def get_expected_ipi_score(
    wrapper: Any,
    input_ids: torch.Tensor,
    option_token_ids: dict[int, list[int]],
    activation_multipliers: Optional[Dict[str, float]] = None,
    intervention_scope: str = DEFAULT_SCOPE,
    last_k: int = DEFAULT_LAST_K,
) -> float:
    last_logits = forward_last_token_logits(
        wrapper=wrapper,
        input_ids=input_ids,
        activation_multipliers=activation_multipliers,
        intervention_scope=intervention_scope,
        last_k=last_k,
    )
    return expected_ipi_from_logits(last_logits, option_token_ids)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ipi_eval import create_ipi_prompt, format_chat_prompt
    from model_factory import get_model_wrapper
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "google/gemma-3-4b-it",
                "wrapper": "gemma",
                "n_devices": 1,
                "dtype": "bfloat16",
            },
            "extraction": {"device": "cpu"},
            "ipi": {"language": "pt"},
        }
    )
    wrapper = get_model_wrapper(cfg, device="cpu")
    tokenizer = wrapper.model.tokenizer
    user_message = create_ipi_prompt("Exemplo de afirmação política.", language="pt")
    prompt = format_chat_prompt(tokenizer, user_message, language="pt")
    option_ids = discover_option_token_ids(tokenizer, prompt)
    for score in SCORES_ORDERED:
        decoded = [tokenizer.decode([tid]) for tid in option_ids[score]]
        print(f"score {score:+d}: token_ids={option_ids[score]} decoded={decoded}")
