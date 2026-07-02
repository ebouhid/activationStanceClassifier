from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

"""Centralized seed resolution: stage override → experiment → global."""


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return value is None


def _coerce_optional_int(value: Any) -> int | None:
    if _is_null(value):
        return None
    return int(value)


def global_seed_from_cfg(cfg: DictConfig | Mapping[str, Any]) -> int:
    """Return the canonical global seed, with one-release random_state fallback."""
    if isinstance(cfg, DictConfig):
        raw_global = cfg.get("seed")
        raw_legacy = cfg.get("random_state")
    else:
        raw_global = cfg.get("seed") if hasattr(cfg, "get") else None
        raw_legacy = cfg.get("random_state") if hasattr(cfg, "get") else None

    if not _is_null(raw_global):
        return int(raw_global)
    if not _is_null(raw_legacy):
        warnings.warn(
            "Top-level random_state is deprecated; set seed: <int> in config.yaml.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(raw_legacy)
    raise ValueError(
        "Global seed is required. Set seed: <int> in config.yaml "
        "(or pass seed=<int> on the CLI)."
    )


def seed_sweep_values(cfg: DictConfig | Mapping[str, Any]) -> list[int | None]:
    """Return the list of seed overrides to sweep over.

    Reads the ``experiment.seeds`` list, the single source of truth for which
    seeds the pipeline runs (use a one-element list for a single run). The list
    of ints is de-duplicated while preserving order. When the field is absent or
    empty the function returns ``[None]``, a single sentinel meaning "use the
    config's global seed" — so callers that simply iterate stay backward
    compatible and pass ``None`` to ``resolve_seeds_from_cfg``.
    """
    experiment = cfg.get("experiment") if hasattr(cfg, "get") else None
    if experiment is None or _is_null(experiment):
        return [None]
    raw = experiment.get("seeds") if hasattr(experiment, "get") else None
    if _is_null(raw):
        return [None]
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"experiment.seeds must be a list of ints when set, got {raw!r}."
        )
    seen: set[int] = set()
    values: list[int | None] = []
    for item in raw:
        value = int(item)
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values or [None]


def resolve_seed(global_seed: int, stage_seed: int | None) -> int:
    """Two-tier resolution: explicit stage seed wins, else the global seed."""
    if stage_seed is not None:
        return int(stage_seed)
    return int(global_seed)


def resolve_nested_seed(parent_resolved: int, stage_seed: int | None) -> int:
    """Nested sub-process: explicit override, else parent's resolved seed."""
    if stage_seed is not None:
        return int(stage_seed)
    return int(parent_resolved)


def _stage_seed(cfg: DictConfig, *keys: str) -> int | None:
    node = cfg
    for key in keys:
        if not hasattr(node, "get"):
            return None
        node = node.get(key)
        if node is None or _is_null(node):
            return None
    return _coerce_optional_int(node)


@dataclass(frozen=True)
class ResolvedSeeds:
    global_seed: int
    training: int
    feature_selection: int
    extraction: int
    optimization: int
    optimization_fast_sample: int
    optimization_split: int
    ipi: int
    poeta: int


def resolve_seeds_from_cfg(
    cfg: DictConfig, seed_override: int | None = None
) -> ResolvedSeeds:
    """Resolve every registered process seed from a composed Hydra config.

    When ``seed_override`` is provided it replaces the canonical global seed for
    this resolution. This is how the pipeline runner threads a single value from
    the ``experiment.seeds`` sweep without mutating the config: the override
    becomes the base that every otherwise-unset stage inherits, and it is also
    what gets threaded to subprocesses as the top-level ``seed=`` override.
    Explicit stage-level seeds still take precedence. ``seed_override=None``
    falls back to the config's global seed.
    """
    if seed_override is not None:
        global_seed = int(seed_override)
    else:
        global_seed = global_seed_from_cfg(cfg)

    training = resolve_seed(
        global_seed,
        _stage_seed(cfg, "training", "random_state"),
    )
    feature_selection = resolve_seed(
        global_seed,
        _stage_seed(cfg, "feature_selection", "seed"),
    )
    extraction = resolve_seed(
        global_seed,
        _stage_seed(cfg, "extraction", "seed"),
    )
    optimization = resolve_seed(
        global_seed,
        _stage_seed(cfg, "optimization", "seed"),
    )
    optimization_fast_sample = resolve_nested_seed(
        optimization,
        _stage_seed(cfg, "optimization", "fast_sample_seed"),
    )
    optimization_split = resolve_seed(
        global_seed,
        _stage_seed(cfg, "optimization", "split_seed"),
    )
    ipi = resolve_seed(global_seed, _stage_seed(cfg, "ipi", "seed"))
    poeta = resolve_seed(global_seed, _stage_seed(cfg, "poeta", "seed"))

    return ResolvedSeeds(
        global_seed=global_seed,
        training=training,
        feature_selection=feature_selection,
        extraction=extraction,
        optimization=optimization,
        optimization_fast_sample=optimization_fast_sample,
        optimization_split=optimization_split,
        ipi=ipi,
        poeta=poeta,
    )


def resolved_seeds_to_dict(resolved: ResolvedSeeds) -> dict[str, int | None]:
    """Audit map for manifests and W&B (resolved ints, not raw YAML nulls)."""
    return {
        "global": resolved.global_seed,
        "training": resolved.training,
        "feature_selection": resolved.feature_selection,
        "extraction": resolved.extraction,
        "optimization": resolved.optimization,
        "optimization_fast_sample": resolved.optimization_fast_sample,
        "optimization_split": resolved.optimization_split,
        "ipi": resolved.ipi,
        "poeta": resolved.poeta,
    }


def log_resolved_seeds(resolved: ResolvedSeeds, prefix: str = "") -> None:
    """Print a compact table of resolved seeds to stdout."""
    header = f"{prefix} resolved seeds:" if prefix else "resolved seeds:"
    print(header)
    for key, value in resolved_seeds_to_dict(resolved).items():
        print(f"  {key}: {value}")


def seed_cli_overrides(resolved: ResolvedSeeds) -> str:
    """Hydra CLI fragment threading resolved seeds into subprocess commands."""
    return (
        f"seed={resolved.global_seed} "
        f"training.random_state={resolved.training} "
        f"feature_selection.seed={resolved.feature_selection} "
        f"extraction.seed={resolved.extraction} "
        f"optimization.seed={resolved.optimization} "
        f"optimization.fast_sample_seed={resolved.optimization_fast_sample} "
        f"optimization.split_seed={resolved.optimization_split} "
        f"ipi.seed={resolved.ipi} "
        f"poeta.seed={resolved.poeta}"
    )


def apply_torch_seed(seed: int, deterministic: bool = True) -> None:
    """Set torch (and optionally cudnn) seeds for reproducibility."""
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def __main__() -> None:
    assert resolve_seed(42, None) == 42
    assert resolve_seed(42, 99) == 99
    assert resolve_nested_seed(99, None) == 99
    assert resolve_nested_seed(99, 7) == 7

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "training": {"random_state": None},
            "feature_selection": {"seed": None},
            "extraction": {"seed": None},
            "optimization": {
                "seed": None,
                "fast_sample_seed": None,
                "split_seed": None,
            },
            "ipi": {"seed": None},
            "poeta": {"seed": None},
        }
    )
    resolved = resolve_seeds_from_cfg(cfg)
    assert resolved.training == 42
    assert resolved.optimization_fast_sample == 42

    cfg_opt = OmegaConf.merge(cfg, {"optimization": {"seed": 99}})
    resolved_opt = resolve_seeds_from_cfg(cfg_opt)
    assert resolved_opt.optimization == 99
    assert resolved_opt.optimization_fast_sample == 99

    cfg_fast = OmegaConf.merge(
        cfg_opt, {"optimization": {"fast_sample_seed": 7}}
    )
    resolved_fast = resolve_seeds_from_cfg(cfg_fast)
    assert resolved_fast.optimization_fast_sample == 7

    assert seed_sweep_values(cfg) == [None]
    cfg_sweep = OmegaConf.merge(cfg, {"experiment": {"seeds": [42, 43, 43, 7]}})
    assert seed_sweep_values(cfg_sweep) == [42, 43, 7]
    assert resolve_seeds_from_cfg(cfg, seed_override=7).optimization == 7
    # Explicit stage seed still wins over a sweep override.
    assert resolve_seeds_from_cfg(cfg_opt, seed_override=7).optimization == 99

    print("seeds.py smoke test: OK")
    log_resolved_seeds(resolved_fast, prefix="[smoke]")


if __name__ == "__main__":
    __main__()
