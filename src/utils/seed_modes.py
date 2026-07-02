from __future__ import annotations

from typing import Any

SEED_MODE_SHORTNAMES = ("conv", "order", "joint")

SEED_MODE_LABELS: dict[str, str] = {
    # Runtime / torch seed only; canonical A–E mapping.
    "conv": "conventional runtime seed only",
    # Alternative letter ordering only; generation uses fixed_runtime_seed.
    "order": "alternative ordering only",
    # Sweep seed drives both runtime and alternative ordering.
    "joint": "joint runtime and alternative ordering",
}


def validate_seed_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SEED_MODE_SHORTNAMES:
        raise ValueError(
            f"Invalid seed_mode={mode!r}. "
            f"Expected one of {SEED_MODE_SHORTNAMES} "
            f"({', '.join(SEED_MODE_LABELS[m] for m in SEED_MODE_SHORTNAMES)})."
        )
    return normalized


def resolve_runtime_seed(
    sweep_seed: int,
    seed_mode: str,
    *,
    fixed_runtime_seed: int = 1,
) -> int:
    """Return the torch / generation seed for one sweep iteration."""
    mode = validate_seed_mode(seed_mode)
    if mode == "order":
        return int(fixed_runtime_seed)
    return int(sweep_seed)


def resolve_mapping_seed(sweep_seed: int, seed_mode: str) -> int | None:
    """Return the letter-ordering seed, or None when mapping stays canonical."""
    mode = validate_seed_mode(seed_mode)
    if mode == "conv":
        return None
    return int(sweep_seed)


def resolve_option_scores_for_mode(
    sweep_seed: int,
    seed_mode: str,
    *,
    language: str = "pt",
) -> dict[str, int]:
    """Letter→score map for one sweep iteration under the chosen seed mode."""
    from utils.ipi_surrogate import IPI_OPTION_SCORES, letter_to_score_from_seed

    mapping_seed = resolve_mapping_seed(sweep_seed, seed_mode)
    if mapping_seed is None:
        return dict(IPI_OPTION_SCORES)
    return letter_to_score_from_seed(mapping_seed, language=language)


def merged_question_score_variance_cfg(cfg: Any) -> dict[str, Any]:
    """Merge ``question_score_variance`` defaults, experiment overrides, and CLI."""
    from omegaconf import DictConfig, OmegaConf

    merged: dict[str, Any] = {}
    if hasattr(cfg, "get"):
        base = cfg.get("question_score_variance")
        if base is not None:
            merged.update(OmegaConf.to_container(base, resolve=True) or {})

        experiment = cfg.get("experiment")
        if experiment is not None and hasattr(experiment, "get"):
            exp_qvar = experiment.get("question_score_variance")
            if exp_qvar is not None:
                merged.update(OmegaConf.to_container(exp_qvar, resolve=True) or {})
            for key in (
                "seed_mode",
                "fixed_runtime_seed",
                "option_mapping_seeds",
                "questions_csv",
                "score_scale",
                "max_questions",
                "output_dir",
            ):
                value = experiment.get(key)
                if value is not None:
                    merged[key] = (
                        OmegaConf.to_container(value, resolve=True)
                        if isinstance(value, DictConfig)
                        else value
                    )

        for key in (
            "seed_mode",
            "fixed_runtime_seed",
            "option_mapping_seeds",
            "questions_csv",
            "score_scale",
            "max_questions",
            "output_dir",
        ):
            value = cfg.get(key)
            if value is not None:
                merged[key] = (
                    OmegaConf.to_container(value, resolve=True)
                    if isinstance(value, DictConfig)
                    else value
                )

    return merged


def qvar_cfg_value(qvar_cfg: dict[str, Any], key: str, default: Any) -> Any:
    value = qvar_cfg.get(key, default)
    if value is None:
        return default
    return value
