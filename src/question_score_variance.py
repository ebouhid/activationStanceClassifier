"""
Baseline per-question score variance across sweep seeds.

Runs discrete IPI evaluation (no interventions) on a question CSV for each
seed in ``option_mapping_seeds``, aggregates mean/variance and mean pairwise
EMD per question, and plots metrics with questions ordered by ascending
variance or EMD, plus a per-seed score scatter using the variance ordering.

Seed modes (``seed_mode`` shortnames):

- ``conv``  — runtime/torch seed only; canonical A–E mapping
- ``order`` — alternative ordering only; ``fixed_runtime_seed`` for generation
- ``joint`` — sweep seed drives both runtime and alternative ordering
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ipi_eval import run_ipi_test
from model_factory import get_model_wrapper
from utils.ipi_surrogate import IPI_OPTION_LETTERS, SCORES_ORDERED
from utils.seed_modes import (
    SEED_MODE_LABELS,
    merged_question_score_variance_cfg,
    qvar_cfg_value,
    resolve_mapping_seed,
    resolve_option_scores_for_mode,
    resolve_runtime_seed,
    validate_seed_mode,
)
from utils.seeds import apply_torch_seed

_SCORE_TO_ALTERNATIVE = {score: alt for alt, score in enumerate(SCORES_ORDERED, start=1)}
_VALID_SCORE_SCALES = frozenset({"alternative", "ipi"})


def _resolve_questions_csv(cfg: DictConfig, qvar_cfg: dict[str, Any]) -> str:
    path = qvar_cfg_value(qvar_cfg, "questions_csv", None)
    if path is None:
        data_cfg = cfg.get("data", {}) or {}
        path = data_cfg.get("validation_dataset")
    if path is None or str(path).strip() == "":
        raise ValueError(
            "questions_csv is required (or set data.validation_dataset in config)."
        )
    return hydra.utils.to_absolute_path(str(path))


def _resolve_sweep_seeds(qvar_cfg: dict[str, Any]) -> list[int]:
    seeds = qvar_cfg_value(qvar_cfg, "option_mapping_seeds", None)
    if seeds is None:
        raise ValueError(
            "option_mapping_seeds is required, e.g. option_mapping_seeds=[42,52,62]"
        )
    resolved = [int(s) for s in seeds]
    if not resolved:
        raise ValueError("option_mapping_seeds must contain at least one seed.")
    return resolved


def _resolve_score_scale(qvar_cfg: dict[str, Any]) -> str:
    scale = str(qvar_cfg_value(qvar_cfg, "score_scale", "alternative")).strip().lower()
    if scale not in _VALID_SCORE_SCALES:
        raise ValueError(
            f"Invalid score_scale={scale!r}. Expected one of {sorted(_VALID_SCORE_SCALES)}."
        )
    return scale


def _resolve_seed_mode(qvar_cfg: dict[str, Any]) -> str:
    return validate_seed_mode(qvar_cfg_value(qvar_cfg, "seed_mode", "joint"))


def _resolve_fixed_runtime_seed(qvar_cfg: dict[str, Any]) -> int:
    return int(qvar_cfg_value(qvar_cfg, "fixed_runtime_seed", 1))


def _ipi_to_display_score(ipi_score: int | None, score_scale: str) -> float | None:
    if ipi_score is None:
        return None
    if score_scale == "ipi":
        return float(ipi_score)
    return float(_SCORE_TO_ALTERNATIVE[int(ipi_score)])


def _extract_response_letter(response: str) -> str | None:
    """Parse the first A–E letter from a model response (same rules as IPI eval)."""
    response = response.strip().strip("\n.")
    response = re.sub(r"^[\s\-\*•]+", "", response)
    response = response.split("\n")[0].strip()
    response = re.sub(r"\s*\(.*$", "", response)
    response = re.sub(r"\.+\s*$", "", response)
    response = response.strip().upper()
    if not response:
        return None
    letter = response[0]
    return letter if letter in IPI_OPTION_LETTERS else None


def _mean_pairwise_emd(scores: list[float]) -> float:
    """Mean all-vs-all pairwise 1D Wasserstein distance (handshake over seeds).

    Each seed contributes one score (point mass). For 1D point masses at ``a``
    and ``b``, the Wasserstein-1 distance equals ``|a - b|``.
    """
    if len(scores) == 0:
        return np.nan
    if len(scores) == 1:
        return 0.0
    total = 0.0
    count = 0
    for i, left in enumerate(scores):
        for right in scores[i + 1 :]:
            total += abs(left - right)
            count += 1
    return total / count


def _question_key(row: pd.Series, row_index: int) -> str:
    parts = [f"row{int(row_index):04d}"]
    if "pair_id" in row and pd.notna(row["pair_id"]):
        parts.append(f"pair{int(row['pair_id'])}")
    if "tipo_pergunta" in row and pd.notna(row["tipo_pergunta"]):
        parts.append(str(row["tipo_pergunta"]).strip())
    return "_".join(parts)


def _aggregate_question_stats(
    questions_df: pd.DataFrame,
    per_seed_records: list[dict[str, Any]],
    score_scale: str,
) -> pd.DataFrame:
    scores_by_key: dict[str, list[float]] = {}
    for record in per_seed_records:
        key = record["question_key"]
        display_score = record["display_score"]
        if display_score is None:
            continue
        scores_by_key.setdefault(key, []).append(float(display_score))

    rows: list[dict[str, Any]] = []
    for row_index, row in questions_df.iterrows():
        key = _question_key(row, int(row_index))
        values = scores_by_key.get(key, [])
        if len(values) == 0:
            mean_score = np.nan
            var_score = np.nan
            mean_emd = np.nan
        elif len(values) == 1:
            mean_score = float(values[0])
            var_score = 0.0
            mean_emd = 0.0
        else:
            mean_score = float(np.mean(values))
            var_score = float(np.var(values, ddof=1))
            mean_emd = _mean_pairwise_emd(values)

        rows.append(
            {
                "question_key": key,
                "row_index": int(row_index),
                "pair_id": row.get("pair_id"),
                "tipo_pergunta": row.get("tipo_pergunta"),
                "eixo": row.get("eixo"),
                "pergunta": row.get("pergunta"),
                "n_valid_scores": len(values),
                "mean_score": mean_score,
                "var_score": var_score,
                "mean_emd": mean_emd,
                "score_scale": score_scale,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values(
        ["var_score", "row_index"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    stats_df["variance_rank"] = np.arange(1, len(stats_df) + 1)

    emd_order = stats_df.sort_values(
        ["mean_emd", "row_index"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    emd_rank = {
        row["question_key"]: int(idx) + 1
        for idx, row in emd_order.iterrows()
    }
    stats_df["emd_rank"] = stats_df["question_key"].map(emd_rank)
    return stats_df


def _plot_metric_by_rank(
    stats_df: pd.DataFrame,
    output_path: Path,
    title: str,
    metric_col: str,
    rank_col: str,
    ylabel: str,
    xlabel: str,
) -> None:
    plot_df = stats_df.sort_values(rank_col, kind="mergesort")
    x = plot_df[rank_col].to_numpy()
    y = plot_df[metric_col].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o", markersize=3, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _score_axis_ticks(score_scale: str) -> tuple[list[float], str]:
    if score_scale == "ipi":
        ticks = [float(v) for v in SCORES_ORDERED]
        return ticks, "Score (-2 to 2)"
    ticks = [float(v) for v in range(1, 6)]
    return ticks, "Score (1 to 5, alternative)"


def _plot_score_scatter(
    stats_df: pd.DataFrame,
    per_seed_df: pd.DataFrame,
    output_path: Path,
    title: str,
    score_scale: str,
) -> None:
    rank_by_key = stats_df.set_index("question_key")["variance_rank"].to_dict()
    y_ticks, y_label = _score_axis_ticks(score_scale)

    seeds = sorted(per_seed_df["sweep_seed"].unique())
    seed_jitter = {
        int(seed): (idx - (len(seeds) - 1) / 2.0) * 0.08
        for idx, seed in enumerate(seeds)
    }

    xs: list[float] = []
    ys: list[float] = []
    for _, row in per_seed_df.iterrows():
        display_score = row.get("display_score")
        if display_score is None or pd.isna(display_score):
            continue
        rank = float(rank_by_key[row["question_key"]])
        xs.append(rank + seed_jitter[int(row["sweep_seed"])])
        ys.append(float(display_score))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(xs, ys, s=18, alpha=0.65, linewidths=0)
    ax.set_yticks(y_ticks)
    ax.set_xlabel("Question (ordered by ascending variance)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
    ax.grid(True, alpha=0.3, axis="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    qvar_cfg = merged_question_score_variance_cfg(cfg)
    seed_mode = _resolve_seed_mode(qvar_cfg)
    fixed_runtime_seed = _resolve_fixed_runtime_seed(qvar_cfg)
    sweep_seeds = _resolve_sweep_seeds(qvar_cfg)
    score_scale = _resolve_score_scale(qvar_cfg)

    questions_path = _resolve_questions_csv(cfg, qvar_cfg)
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    ipi_cfg = dict(cfg.get("ipi", {}) or {})
    language = str(ipi_cfg.get("language", "pt"))
    max_new_tokens = int(ipi_cfg.get("max_new_tokens", 10))
    temperature = float(ipi_cfg.get("temperature", 0.0))
    max_questions = qvar_cfg_value(qvar_cfg, "max_questions", None)
    if max_questions is not None:
        max_questions = int(max_questions)

    model_cfg = cfg.get("model", {}) or {}
    model_name = str(model_cfg.get("name", "unknown-model"))
    if model_cfg.get("name") is None:
        raise ValueError("model.name is required. Pass model=<config-name> on the CLI.")

    output_dir = Path(
        hydra.utils.to_absolute_path(
            str(qvar_cfg_value(qvar_cfg, "output_dir", "runs/question_score_variance"))
        )
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{Path(questions_path).stem}_{seed_mode}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    questions_df = pd.read_csv(questions_path)
    if "pergunta" not in questions_df.columns:
        raise ValueError(f"questions CSV must contain a 'pergunta' column: {questions_path}")
    if max_questions is not None:
        questions_df = questions_df.head(max_questions).copy()

    print(f"Loading model: {model_name}")
    if seed_mode == "order":
        print(
            f"Seed mode: {seed_mode} ({SEED_MODE_LABELS[seed_mode]}); "
            f"runtime_seed fixed at {fixed_runtime_seed}"
        )
    else:
        print(
            f"Seed mode: {seed_mode} ({SEED_MODE_LABELS[seed_mode]}); "
            f"runtime_seed follows each sweep seed"
        )
    wrapper = get_model_wrapper(cfg)

    per_seed_records: list[dict[str, Any]] = []
    for sweep_seed in sweep_seeds:
        runtime_seed = resolve_runtime_seed(
            sweep_seed,
            seed_mode,
            fixed_runtime_seed=fixed_runtime_seed,
        )
        mapping_seed = resolve_mapping_seed(sweep_seed, seed_mode)
        option_scores = resolve_option_scores_for_mode(
            sweep_seed,
            seed_mode,
            language=language,
        )
        apply_torch_seed(runtime_seed)
        print(
            f"\nRunning baseline IPI for sweep_seed={sweep_seed} "
            f"(runtime_seed={runtime_seed}, mapping_seed={mapping_seed}) "
            f"— {len(questions_df)} questions..."
        )
        results_df = run_ipi_test(
            wrapper=wrapper,
            questions_df=questions_df,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            activation_multipliers=None,
            verbose=True,
            option_scores=option_scores,
            transcripts_dir=None,
        )

        if len(results_df) != len(questions_df):
            raise RuntimeError(
                f"Result row count ({len(results_df)}) != question count ({len(questions_df)}) "
                f"for sweep_seed={sweep_seed}"
            )

        for i, (row_index, row) in enumerate(questions_df.iterrows()):
            result_row = results_df.iloc[i]
            key = _question_key(row, int(row_index))
            ipi_score = result_row["ipi_score"]
            if pd.isna(ipi_score):
                ipi_score = None
            else:
                ipi_score = int(ipi_score)

            per_seed_records.append(
                {
                    "seed_mode": seed_mode,
                    "sweep_seed": int(sweep_seed),
                    "runtime_seed": int(runtime_seed),
                    "mapping_seed": mapping_seed,
                    "question_key": key,
                    "row_index": int(row_index),
                    "pair_id": row.get("pair_id"),
                    "tipo_pergunta": row.get("tipo_pergunta"),
                    "eixo": row.get("eixo"),
                    "pergunta": row.get("pergunta"),
                    "response_letter": _extract_response_letter(
                        str(result_row["model_response_raw"])
                    ),
                    "ipi_score": ipi_score,
                    "display_score": _ipi_to_display_score(ipi_score, score_scale),
                    "model_response_raw": result_row["model_response_raw"],
                }
            )

    stats_df = _aggregate_question_stats(questions_df, per_seed_records, score_scale)
    per_seed_df = pd.DataFrame(per_seed_records)

    stats_csv = run_dir / "question_stats.csv"
    per_seed_csv = run_dir / "per_seed_scores.csv"
    plot_path = run_dir / "variance_by_question.png"
    emd_plot_path = run_dir / "emd_by_question.png"
    scatter_path = run_dir / "score_scatter_by_question.png"
    meta_path = run_dir / "run_metadata.json"

    stats_df.to_csv(stats_csv, index=False)
    per_seed_df.to_csv(per_seed_csv, index=False)

    scale_label = "1-5 (alternative)" if score_scale == "alternative" else "-2 to 2 (ipi)"
    seed_summary = (
        f"mode={seed_mode}, sweep={sweep_seeds}, "
        f"fixed_runtime={fixed_runtime_seed}"
    )
    plot_title = (
        f"{model_name} — baseline score variance\n"
        f"scale={scale_label}, {seed_summary}"
    )
    scatter_title = (
        f"{model_name} — baseline scores per seed\n"
        f"scale={scale_label}, {seed_summary}"
    )
    emd_plot_title = (
        f"{model_name} — mean pairwise EMD across seeds\n"
        f"scale={scale_label}, {seed_summary}"
    )
    _plot_metric_by_rank(
        stats_df,
        plot_path,
        plot_title,
        metric_col="var_score",
        rank_col="variance_rank",
        ylabel="Score variance",
        xlabel="Question (ordered by ascending variance)",
    )
    _plot_metric_by_rank(
        stats_df,
        emd_plot_path,
        emd_plot_title,
        metric_col="mean_emd",
        rank_col="emd_rank",
        ylabel="Mean pairwise EMD",
        xlabel="Question (ordered by ascending mean EMD)",
    )
    _plot_score_scatter(
        stats_df, per_seed_df, scatter_path, scatter_title, score_scale
    )

    metadata = {
        "model_name": model_name,
        "questions_csv": questions_path,
        "n_questions": int(len(questions_df)),
        "seed_mode": seed_mode,
        "seed_mode_label": SEED_MODE_LABELS[seed_mode],
        "fixed_runtime_seed": fixed_runtime_seed,
        "option_mapping_seeds": sweep_seeds,
        "score_scale": score_scale,
        "language": language,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "output_dir": str(run_dir),
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote question stats: {stats_csv}")
    print(f"Wrote per-seed scores: {per_seed_csv}")
    print(f"Wrote variance plot: {plot_path}")
    print(f"Wrote mean EMD plot: {emd_plot_path}")
    print(f"Wrote score scatter plot: {scatter_path}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
