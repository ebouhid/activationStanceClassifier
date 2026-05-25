from __future__ import annotations

import logging
import random
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Mapping, Optional

_logger = logging.getLogger(__name__)
_last_option_scores_log: dict[str, Any] | None = None

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


def option_scores_from_seed(seed: int) -> dict[str, int]:
    """Deterministic permutation of scores across A–E for a given seed.

    The same mapping applies to every question in a run; different seeds
    produce different letter→score assignments (each score in {-2..2}
    appears exactly once).
    """
    scores = list(SCORES_ORDERED)
    rng = random.Random(int(seed))
    rng.shuffle(scores)
    mapping = {letter: score for letter, score in zip(IPI_OPTION_LETTERS, scores)}
    log_option_scores_mapping(
        mapping,
        source="option_scores_from_seed",
        seed=int(seed),
    )
    return mapping


def format_option_scores(option_scores: Mapping[str, int]) -> str:
    return ", ".join(
        f"{letter}={option_scores[letter]:+d}" for letter in IPI_OPTION_LETTERS
    )


def option_scores_to_alternative_numbers(
    option_scores: Mapping[str, int],
) -> dict[int, int]:
    """Map alternative index 1–5 (A–E) to IPI score."""
    return {
        alt: int(option_scores[letter])
        for alt, letter in enumerate(IPI_OPTION_LETTERS, start=1)
    }


def format_option_scores_alternative(option_scores: Mapping[str, int]) -> str:
    alt_map = option_scores_to_alternative_numbers(option_scores)
    return ", ".join(f"{alt}={alt_map[alt]:+d}" for alt in sorted(alt_map))


def build_option_scores_log_payload(
    option_scores: Mapping[str, int],
    *,
    source: str,
    seed: int | None = None,
) -> dict[str, Any]:
    letter_map = {letter: int(option_scores[letter]) for letter in IPI_OPTION_LETTERS}
    alt_map = option_scores_to_alternative_numbers(option_scores)
    payload: dict[str, Any] = {
        "option_scores_source": source,
        "option_scores_letter": letter_map,
        "option_scores_alternative": alt_map,
        "option_scores_letter_str": format_option_scores(letter_map),
        "option_scores_alternative_str": format_option_scores_alternative(letter_map),
    }
    if seed is not None:
        payload["option_mapping_seed"] = int(seed)
    return payload


def log_option_scores_mapping(
    option_scores: Mapping[str, int],
    *,
    source: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Log letter and alternative-number mappings (terminal + W&B when active)."""
    global _last_option_scores_log
    payload = build_option_scores_log_payload(
        option_scores, source=source, seed=seed
    )
    _last_option_scores_log = payload
    seed_part = f" (seed={seed})" if seed is not None else ""
    message = (
        f"IPI option scores [{source}]{seed_part}: "
        f"letters: {payload['option_scores_letter_str']}; "
        f"alternatives: {payload['option_scores_alternative_str']}"
    )
    print(message)
    _logger.info(message)
    flush_option_scores_wandb_log()
    return payload


def flush_option_scores_wandb_log() -> bool:
    """Push the last option-score mapping to the active W&B run, if any."""
    if _last_option_scores_log is None:
        return False
    try:
        import wandb
    except ImportError:
        return False
    if wandb.run is None:
        return False
    wandb.config.update(_last_option_scores_log, allow_val_change=True)
    wandb.summary.update(_last_option_scores_log)
    return True


def seed_dependent_option_scores_enabled(cfg: Any) -> bool:
    ipi_cfg = cfg.get("ipi", {}) if hasattr(cfg, "get") else {}
    return bool(ipi_cfg.get("seed_dependent_option_scores", False))


def resolve_option_mapping_seed(cfg: Any) -> int:
    """Seed used for letter→score permutation when seed-dependent mapping is on."""
    from utils.seeds import _stage_seed, resolve_seeds_from_cfg

    explicit = _stage_seed(cfg, "ipi", "option_mapping_seed")
    if explicit is not None:
        return int(explicit)
    return int(resolve_seeds_from_cfg(cfg).ipi)


def resolve_option_scores(cfg: Any) -> dict[str, int]:
    """Letter→score map for this Hydra config (canonical or seed-permuted)."""
    if not seed_dependent_option_scores_enabled(cfg):
        mapping = dict(IPI_OPTION_SCORES)
        log_option_scores_mapping(mapping, source="resolve_option_scores")
        return mapping
    return option_scores_from_seed(resolve_option_mapping_seed(cfg))


def option_letter_variants(letter: str) -> list[str]:
    return [letter, f" {letter}", f"\n{letter}", f"{letter}.", f"{letter})"]


def discover_option_token_ids(
    tokenizer: Any,
    prompt_text: str,
    option_scores: Mapping[str, int] | None = None,
) -> dict[int, list[int]]:
    """Discover single-token IDs for A–E answers in chat-template context."""
    scores_map = dict(option_scores or IPI_OPTION_SCORES)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    option_ids: dict[int, list[int]] = {}

    for letter in IPI_OPTION_LETTERS:
        score = scores_map[letter]
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
    for seed in (42, 43, 44):
        permuted = option_scores_from_seed(seed)
        print(f"seed {seed}: {format_option_scores(permuted)}")

    option_ids = discover_option_token_ids(tokenizer, prompt)
    for score in SCORES_ORDERED:
        decoded = [tokenizer.decode([tid]) for tid in option_ids[score]]
        print(f"score {score:+d}: token_ids={option_ids[score]} decoded={decoded}")
