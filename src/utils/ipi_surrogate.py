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

# Immutable text↔score semantics (Likert order: most disagree → most agree).
IPI_OPTION_TEXT: dict[str, dict[int, str]] = {
    "pt": {
        -2: "Discordo totalmente",
        -1: "Discordo",
        0: "Nem concordo nem discordo",
        1: "Concordo",
        2: "Concordo totalmente",
    },
    "en": {
        -2: "Strongly disagree",
        -1: "Disagree",
        0: "Neither agree nor disagree",
        1: "Agree",
        2: "Strongly agree",
    },
}


def score_to_letter_map(letter_to_score: Mapping[str, int]) -> dict[int, str]:
    return {int(score): letter for letter, score in letter_to_score.items()}


def letter_to_score_from_seed(seed: int, language: str = "pt") -> dict[str, int]:
    """Shuffle which letter labels each canonical Likert text (score fixed).

    Text→score never changes (e.g. "Discordo totalmente" → -2). Seed only
    permutes letter→text/score. Prompts list options in letter order A–E with
    permuted lines, e.g. ``B. Discordo totalmente`` when B maps to -2.
    """
    letters = list(IPI_OPTION_LETTERS)
    rng = random.Random(int(seed))
    rng.shuffle(letters)
    mapping = {
        letter: score for letter, score in zip(letters, SCORES_ORDERED)
    }
    log_option_scores_mapping(
        mapping,
        source="letter_to_score_from_seed",
        seed=int(seed),
        language=language,
    )
    return mapping


def option_scores_from_seed(seed: int, language: str = "pt") -> dict[str, int]:
    """Alias for :func:`letter_to_score_from_seed` (letter→score map)."""
    return letter_to_score_from_seed(seed, language=language)


def format_ipi_options_block(
    language: str,
    letter_to_score: Mapping[str, int] | None = None,
) -> str:
    """Build the five options in letter order A–E (text per seed mapping)."""
    if language not in IPI_OPTION_TEXT:
        raise ValueError(
            f"Unsupported IPI prompt language {language!r}. Expected 'pt' or 'en'."
        )
    scores_map = dict(letter_to_score or IPI_OPTION_SCORES)
    return "\n".join(
        f"{letter}. {IPI_OPTION_TEXT[language][scores_map[letter]]}"
        for letter in IPI_OPTION_LETTERS
    )


def format_letter_to_text(
    letter_to_score: Mapping[str, int],
    language: str = "pt",
) -> str:
    """Compact letter→Likert text for logging."""
    scores_map = dict(letter_to_score)
    return ", ".join(
        f"{letter}={IPI_OPTION_TEXT[language][scores_map[letter]]!r}"
        for letter in IPI_OPTION_LETTERS
    )


def format_option_scores(option_scores: Mapping[str, int]) -> str:
    return ", ".join(
        f"{letter}={option_scores[letter]:+d}" for letter in IPI_OPTION_LETTERS
    )


def canonical_alternative_scores() -> dict[int, int]:
    """Semantic alternative index 1–5 (Likert order) → fixed IPI score."""
    return {alt: SCORES_ORDERED[alt - 1] for alt in range(1, 6)}


def format_score_to_letter(score_to_letter: Mapping[int, str]) -> str:
    return ", ".join(
        f"{score:+d}={score_to_letter[score]}" for score in SCORES_ORDERED
    )


def format_canonical_text_scores(language: str = "pt") -> str:
    return ", ".join(
        f"{alt}({SCORES_ORDERED[alt - 1]:+d})={IPI_OPTION_TEXT[language][SCORES_ORDERED[alt - 1]]}"
        for alt in range(1, 6)
    )


def format_option_scores_alternative() -> str:
    alt_map = canonical_alternative_scores()
    return ", ".join(f"{alt}={alt_map[alt]:+d}" for alt in sorted(alt_map))


def build_option_scores_log_payload(
    option_scores: Mapping[str, int],
    *,
    source: str,
    seed: int | None = None,
    language: str = "pt",
) -> dict[str, Any]:
    letter_map = {letter: int(option_scores[letter]) for letter in IPI_OPTION_LETTERS}
    score_letter = score_to_letter_map(letter_map)
    alt_map = canonical_alternative_scores()
    payload: dict[str, Any] = {
        "option_scores_source": source,
        "option_scores_letter": letter_map,
        "option_score_to_letter": score_letter,
        "option_scores_alternative": alt_map,
        "option_scores_letter_str": format_option_scores(letter_map),
        "option_score_to_letter_str": format_score_to_letter(score_letter),
        "option_letter_to_text_str": format_letter_to_text(letter_map, language),
        "option_scores_alternative_str": format_option_scores_alternative(),
        "option_text_scores_canonical_str": format_canonical_text_scores(language),
    }
    if seed is not None:
        payload["option_mapping_seed"] = int(seed)
    return payload


def log_option_scores_mapping(
    option_scores: Mapping[str, int],
    *,
    source: str,
    seed: int | None = None,
    language: str = "pt",
) -> dict[str, Any]:
    """Log canonical text→score and seed-dependent score→letter maps."""
    global _last_option_scores_log
    payload = build_option_scores_log_payload(
        option_scores, source=source, seed=seed, language=language
    )
    _last_option_scores_log = payload
    seed_part = f" (seed={seed})" if seed is not None else ""
    message = (
        f"IPI option mapping [{source}]{seed_part}: "
        f"text→score (fixed): {payload['option_text_scores_canonical_str']}; "
        f"letter→text: {payload['option_letter_to_text_str']}; "
        f"letter→score: {payload['option_scores_letter_str']}"
    )
    print(message)
    _logger.info(message)
    flush_option_scores_wandb_log()
    return payload


def _wandb_safe_option_scores_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """W&B config/summary encoding requires string keys in nested dicts."""
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            safe[key] = {str(k): v for k, v in value.items()}
        else:
            safe[key] = value
    return safe


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
    safe_payload = _wandb_safe_option_scores_payload(_last_option_scores_log)
    wandb.config.update(safe_payload, allow_val_change=True)
    wandb.summary.update(safe_payload)
    return True


def _ipi_cfg(cfg: Any) -> dict[str, Any]:
    """Top-level ``ipi`` block as a plain dict (experiment overrides are threaded
    explicitly by the pipeline orchestrator, so no cross-namespace merge needed)."""
    from omegaconf import OmegaConf

    if hasattr(cfg, "get"):
        base = cfg.get("ipi")
        if base is not None:
            return OmegaConf.to_container(base, resolve=True) or {}
    return {}


def seed_dependent_option_scores_enabled(cfg: Any) -> bool:
    return bool(_ipi_cfg(cfg).get("seed_dependent_option_scores", False))


def resolve_option_mapping_seed(cfg: Any) -> int:
    """Seed used for letter↔text permutation when seed-dependent mapping is on."""
    from utils.seeds import _stage_seed, resolve_seeds_from_cfg

    explicit = _stage_seed(cfg, "ipi", "option_mapping_seed")
    if explicit is not None:
        return int(explicit)
    return int(resolve_seeds_from_cfg(cfg).ipi)


def resolve_option_scores(cfg: Any) -> dict[str, int]:
    """Letter→score map for this Hydra config (canonical or seed-shuffled letters)."""
    ipi_cfg = _ipi_cfg(cfg)
    language = str(ipi_cfg.get("language", "pt"))
    if not seed_dependent_option_scores_enabled(cfg):
        mapping = dict(IPI_OPTION_SCORES)
        log_option_scores_mapping(
            mapping, source="resolve_option_scores", language=language
        )
        return mapping
    mapping_seed = resolve_option_mapping_seed(cfg)
    return letter_to_score_from_seed(mapping_seed, language=language)


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
    for seed in (42, 43, 44):
        permuted = letter_to_score_from_seed(seed)
        print(f"seed {seed} options block:\n{format_ipi_options_block('pt', permuted)}")
    user_message = create_ipi_prompt(
        "Exemplo de afirmação política.", language="pt", option_scores=permuted
    )
    prompt = format_chat_prompt(tokenizer, user_message, language="pt")

    option_ids = discover_option_token_ids(tokenizer, prompt)
    for score in SCORES_ORDERED:
        decoded = [tokenizer.decode([tid]) for tid in option_ids[score]]
        print(f"score {score:+d}: token_ids={option_ids[score]} decoded={decoded}")
