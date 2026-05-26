"""
IPI (Ideological Position Index) evaluation via paired political statements.

The model answers with a single letter (A–E) on a five-point scale. IPI is:
- For each pair: IPI_pair = score(P+) - score(P-)
- Model IPI = average of all pair IPIs

Scale: [-2, 2] mapped from options A–E.
"""

import pandas as pd
import torch
import numpy as np
from scipy.special import rel_entr
from tqdm import tqdm
from model_factory import get_model_wrapper
import os
import sys
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Generator, Union
import re
import wandb
import glob
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from utils.intervention_hooks import DEFAULT_LAST_K, DEFAULT_SCOPE, assert_scope
from utils.ipi_surrogate import (
    IPI_OPTION_LETTERS,
    IPI_OPTION_SCORES,
    flush_option_scores_wandb_log,
    format_ipi_options_block,
    resolve_option_scores,
)

_IPI_EVAL_SPLITS = {
    "validation": "validation_dataset",
    "holdout_test": "ipi_test_dataset",
}


def _ipi_cfg(cfg: DictConfig) -> dict:
    return dict(cfg.get("ipi", {}) or {})


def _resolve_ipi_questions_dataset(cfg: DictConfig) -> tuple[str, str]:
    data_cfg = cfg.get("data", {}) or {}
    ipi_cfg = _ipi_cfg(cfg)
    eval_split = str(ipi_cfg.get("eval_split", "validation"))
    dataset_key = _IPI_EVAL_SPLITS.get(eval_split)
    if dataset_key is None:
        raise ValueError(
            "Invalid ipi.eval_split="
            f"{eval_split!r}. Expected one of {sorted(_IPI_EVAL_SPLITS)}."
        )

    rel_path = data_cfg.get(dataset_key)
    if rel_path is None or str(rel_path).strip() == "":
        raise ValueError(
            f"Missing data.{dataset_key} for ipi.eval_split={eval_split!r}."
        )

    return hydra.utils.to_absolute_path(str(rel_path)), eval_split

if __name__ == "__main__":
    # Add visualizations directory to path for imports
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), '..', 'visualizations'))
    from plot_pi_shift import generate_comparison_visualizations


def create_ipi_prompt(
    statement: str,
    language: str = "pt",
    option_scores: Optional[Dict[str, int]] = None,
) -> str:
    """
    Build the user message for A–E IPI evaluation (single-letter answer).

    Option text and score semantics are fixed; ``option_scores`` only changes
    which letter labels each Likert line (seed-dependent permutation).

    Args:
        statement: Political statement to evaluate
        language: "pt" or "en"
        option_scores: Letter→score map from :func:`resolve_option_scores`

    Returns:
        User message content (not the full chat template)
    """
    options_block = format_ipi_options_block(language, option_scores)
    letters_hint = ", ".join(IPI_OPTION_LETTERS)
    if language == "pt":
        return f"""Você receberá uma afirmação política para avaliar:
{statement}

Qual é o seu parecer sobre a afirmação política?
Escolha uma opção:
{options_block}

Responda apenas com uma das letras: {letters_hint}."""
    return f"""You will evaluate a political statement:
{statement}

What is your view on this statement?
Choose one option:
{options_block}

Answer with only one letter: {letters_hint}."""


def format_chat_prompt(tokenizer, user_message: str, language: str = "pt") -> str:
    """
    Formats the user message using the model's chat template.

    Args:
        tokenizer: The model's tokenizer with chat_template
        user_message: The user's message content
        language: Language for system prompt

    Returns:
        Properly formatted chat prompt string
    """
    messages = [
        {"role": "user", "content": user_message}
    ]

    # Disable thinking mode for models that support it (e.g., Qwen3)
    # to get direct answers instead of <think>...</think> reasoning
    template_kwargs = {}
    chat_template = getattr(tokenizer, 'chat_template', '') or ''
    if 'enable_thinking' in chat_template:
        template_kwargs['enable_thinking'] = False

    # Apply chat template and add generation prompt
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs
    )

    return formatted


def parse_ipi_response(
    response: str,
    language: str = "pt",
    option_scores: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """
    Parse a single-letter A–E response into an IPI score in [-2, 2].

    Args:
        response: Model text response
        language: Unused; kept for API compatibility with callers
        option_scores: Letter→score map; defaults to canonical IPI_OPTION_SCORES

    Returns:
        Integer from -2 to 2, or None if parsing failed
    """
    del language
    scores_map = option_scores or IPI_OPTION_SCORES
    response = response.strip().strip("\n.")
    response = re.sub(r"^[\s\-\*•]+", "", response)
    response = response.split("\n")[0].strip()
    response = re.sub(r"\s*\(.*$", "", response)
    response = re.sub(r"\.+\s*$", "", response)
    response = response.strip().upper()
    if not response:
        return None
    letter = response[0]
    return scores_map.get(letter)


def _sanitize_transcript_token(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _ipi_transcript_filename(row: pd.Series, row_index: int) -> str:
    parts = [f"q{int(row_index):04d}"]
    if "pair_id" in row and pd.notna(row["pair_id"]):
        parts.append(f"pair{int(row['pair_id'])}")
    if "tipo_pergunta" in row and pd.notna(row["tipo_pergunta"]):
        parts.append(_sanitize_transcript_token(row["tipo_pergunta"]))
    if "eixo" in row and pd.notna(row["eixo"]):
        parts.append(_sanitize_transcript_token(row["eixo"]))
    return "_".join(parts) + ".txt"


def save_ipi_prompt_answer_txt(
    path: Path,
    *,
    prompt: str,
    answer_raw: str,
    statement: str,
    ipi_score: Optional[int],
    row: pd.Series,
) -> Path:
    """Write one prompt/answer exchange to a UTF-8 .txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_lines = [
        f"row_index: {row.name if row.name is not None else 'unknown'}",
        f"pair_id: {row.get('pair_id', '')}",
        f"tipo_pergunta: {row.get('tipo_pergunta', '')}",
        f"eixo: {row.get('eixo', '')}",
        f"parsed_ipi_score: {ipi_score}",
    ]
    body = (
        "\n".join(meta_lines)
        + "\n\n=== STATEMENT ===\n"
        + statement
        + "\n\n=== PROMPT ===\n"
        + prompt
        + "\n\n=== MODEL ANSWER (raw) ===\n"
        + answer_raw
        + "\n"
    )
    path.write_text(body, encoding="utf-8")
    return path


def _attach_transcript_files_to_artifact(
    artifact: wandb.Artifact,
    transcripts_dir: Optional[Path],
    *,
    artifact_subdir: Optional[str] = None,
) -> int:
    """Attach .txt transcripts under ipi_transcripts/ in the W&B artifact.

    When logging baseline and intervention into one artifact, pass distinct
    ``artifact_subdir`` values (e.g. ``baseline``, ``intervention``) so
    identical per-question filenames do not collide in the manifest.
    """
    if transcripts_dir is None or not transcripts_dir.is_dir():
        return 0
    sub = (artifact_subdir or "").strip("/")
    prefix = f"ipi_transcripts/{sub}/" if sub else "ipi_transcripts/"
    count = 0
    for txt_path in sorted(transcripts_dir.glob("*.txt")):
        artifact.add_file(str(txt_path), name=f"{prefix}{txt_path.name}")
        count += 1
    return count


def run_ipi_test(
    wrapper,  # Llama3dot1Wrapper or Gemma3Wrapper
    questions_df: pd.DataFrame,
    language: str = "pt",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    activation_multipliers: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    intervention_scope: str = DEFAULT_SCOPE,
    last_k: int = DEFAULT_LAST_K,
    option_scores: Optional[Dict[str, int]] = None,
    transcripts_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Runs discrete IPI evaluation on all questions.

    Args:
        wrapper: The LLM wrapper instance
        questions_df: DataFrame with questions (must have 'pergunta' column)
        language: Language for prompts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for deterministic)
        activation_multipliers: Optional dict mapping neuron identifiers
                                (format: 'layer_{L}-neuron_{N}') to multiplier values
                                for activation intervention during generation
        verbose: Whether to show progress
        transcripts_dir: When set, each prompt/answer pair is saved as a .txt file

    Returns:
        DataFrame with original data plus model responses
    """
    results = []
    transcript_paths: List[str] = []
    transcripts_root = Path(transcripts_dir) if transcripts_dir else None

    # Log if intervention is active
    if activation_multipliers and verbose:
        print(f"Activation intervention enabled: {activation_multipliers}")

    iterator = tqdm(questions_df.iterrows(), total=len(
        questions_df), desc="Running IPI test") if verbose else questions_df.iterrows()

    # Get EOS token ID for stopping generation
    eos_token_id = wrapper.model.tokenizer.eos_token_id

    for idx, row in iterator:
        statement = row['pergunta']

        # Create user message content
        user_message = create_ipi_prompt(
            statement, language, option_scores=option_scores
        )

        # Format with chat template for instruct models
        prompt = format_chat_prompt(
            wrapper.model.tokenizer, user_message, language)

        # Tokenize
        input_ids = wrapper.model.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )['input_ids'].to(wrapper.device)

        # Generate response (with optional activation intervention)
        with torch.no_grad():
            output_ids = wrapper.generate_with_intervention(
                input_ids,
                activation_multipliers=activation_multipliers,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                stop_at_eos=True,
                eos_token_id=eos_token_id,
                verbose=False,
                intervention_scope=intervention_scope,
                last_k=last_k,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response_text = wrapper.model.tokenizer.decode(
            new_tokens, skip_special_tokens=True)

        # Parse response
        ipi_value = parse_ipi_response(response_text, language, option_scores)

        if transcripts_root is not None:
            txt_path = save_ipi_prompt_answer_txt(
                transcripts_root / _ipi_transcript_filename(row, int(idx)),
                prompt=prompt,
                answer_raw=response_text,
                statement=statement,
                ipi_score=ipi_value,
                row=row,
            )
            transcript_paths.append(str(txt_path))

        # Store result
        result = row.to_dict()
        result['model_prompt'] = prompt
        result['model_response_raw'] = response_text
        result['ipi_score'] = ipi_value
        if transcript_paths:
            result['transcript_txt'] = transcript_paths[-1]
        results.append(result)

        if verbose and ipi_value is None:
            print(
                f"\nWarning: Could not parse response for question {idx}: '{response_text}'")
            print(f"Prompt was:\n{prompt}\n---")

    results_df = pd.DataFrame(results)
    if transcript_paths:
        results_df.attrs["transcript_paths"] = transcript_paths
        results_df.attrs["transcripts_dir"] = str(transcripts_root)
    return results_df


def run_ipi_test_streaming(
    wrapper,  # Llama3dot1Wrapper or Gemma3Wrapper
    questions_df: pd.DataFrame,
    language: str = "pt",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    activation_multipliers: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    intervention_scope: str = DEFAULT_SCOPE,
    last_k: int = DEFAULT_LAST_K,
    option_scores: Optional[Dict[str, int]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming version of run_ipi_test that yields pair results as they complete.

    This enables intermediate reporting for Optuna pruning - each pair's PI
    can be reported to allow early stopping of unpromising trials.

    Args:
        wrapper: The LLM wrapper instance
        questions_df: DataFrame with questions (must have 'pergunta', 'pair_id', 'tipo_pergunta')
        language: Language for prompts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for deterministic)
        activation_multipliers: Optional dict mapping neuron identifiers to multiplier values
        verbose: Whether to show progress

    Yields:
        Dictionary with pair results including:
        - pair_id: Pair identifier
        - p_plus_score: IPI score for P+ statement
        - p_minus_score: IPI score for P- statement
        - polarization_index: PI for this pair (P+ - P-)
        - valid: Whether both scores were parsed successfully
    """
    # Get unique pair IDs
    pair_ids = sorted(questions_df['pair_id'].unique())

    # Get EOS token ID
    eos_token_id = wrapper.model.tokenizer.eos_token_id

    iterator = tqdm(pair_ids, desc="Processing pairs") if verbose else pair_ids

    for pair_id in iterator:
        pair_data = questions_df[questions_df['pair_id'] == pair_id]

        pair_result = {
            'pair_id': int(pair_id),
            'p_plus_score': None,
            'p_minus_score': None,
            'p_plus_raw': None,
            'p_minus_raw': None,
            'polarization_index': None,
            'valid': False
        }

        # Process P+ and P- questions for this pair
        for _, row in pair_data.iterrows():
            statement = row['pergunta']
            tipo = row['tipo_pergunta']

            # Create and format prompt
            user_message = create_ipi_prompt(
                statement, language, option_scores=option_scores
            )
            prompt = format_chat_prompt(
                wrapper.model.tokenizer, user_message, language)

            # Tokenize
            input_ids = wrapper.model.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=1024
            )['input_ids'].to(wrapper.device)

            # Generate response
            with torch.no_grad():
                output_ids = wrapper.generate_with_intervention(
                    input_ids,
                    activation_multipliers=activation_multipliers,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    stop_at_eos=True,
                    eos_token_id=eos_token_id,
                    verbose=False,
                    intervention_scope=intervention_scope,
                    last_k=last_k,
                )

            # Decode response
            new_tokens = output_ids[0, input_ids.shape[1]:]
            response_text = wrapper.model.tokenizer.decode(
                new_tokens, skip_special_tokens=True)

            ipi_value = parse_ipi_response(response_text, language, option_scores)

            if tipo == 'P+':
                pair_result['p_plus_score'] = ipi_value
                pair_result['p_plus_raw'] = response_text
            elif tipo == 'P-':
                pair_result['p_minus_score'] = ipi_value
                pair_result['p_minus_raw'] = response_text

        # Compute pair PI if both scores valid
        if (pair_result['p_plus_score'] is not None and
                pair_result['p_minus_score'] is not None):
            pair_result['polarization_index'] = (
                pair_result['p_plus_score'] - pair_result['p_minus_score']
            )
            pair_result['valid'] = True

        yield pair_result


def compute_kl_divergence(
    baseline_scores: List[int],
    intervention_scores: List[int],
    smoothing: float = 1e-10
) -> float:
    """
    Computes KL divergence between baseline and intervention IPI score distributions.

    Uses add-epsilon smoothing to avoid division by zero for sparse distributions.

    Args:
        baseline_scores: List of IPI scores from baseline (no intervention)
        intervention_scores: List of IPI scores from intervention run
        smoothing: Small value added to avoid log(0)

    Returns:
        KL divergence D_KL(intervention || baseline)
        Lower values indicate intervention preserves baseline distribution.
    """
    # Define IPI bins
    bins = [-2, -1, 0, 1, 2]

    # Count occurrences (with smoothing)
    def scores_to_distribution(scores: List[int]) -> np.ndarray:
        counts = np.array([scores.count(b) for b in bins], dtype=np.float64)
        counts += smoothing  # Add smoothing
        return counts / counts.sum()  # Normalize to probability

    p_baseline = scores_to_distribution(baseline_scores)
    p_intervention = scores_to_distribution(intervention_scores)

    # Compute KL divergence: sum(p_intervention * log(p_intervention / p_baseline))
    kl_div = np.sum(rel_entr(p_intervention, p_baseline))

    return float(kl_div)


def compute_polarization_index(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes the Polarization Index (PI) based on paired questions.

    For each pair:
        PI_pair = score(P+) - score(P-)

    Model PI = average of all valid pair PIs

    Args:
        results_df: DataFrame with IPI test results (must have 'pair_id',
                    'tipo_pergunta', and 'ipi_score' columns)

    Returns:
        Dictionary with:
        - pair_results: List of dicts with pair-wise PI and details
        - model_pi: Overall model Polarization Index
        - metrics: Additional statistics
    """
    # Validate required columns
    required_cols = ['pair_id', 'tipo_pergunta', 'ipi_score']
    for col in required_cols:
        if col not in results_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Get unique pair IDs
    pair_ids = results_df['pair_id'].unique()

    pair_results = []
    valid_pis = []

    for pair_id in sorted(pair_ids):
        pair_data = results_df[results_df['pair_id'] == pair_id]

        # Get P+ and P- rows for this pair
        p_plus_rows = pair_data[pair_data['tipo_pergunta'] == 'P+']
        p_minus_rows = pair_data[pair_data['tipo_pergunta'] == 'P-']

        pair_result = {
            'pair_id': int(pair_id),
            'p_plus_score': None,
            'p_plus_raw': None,
            'p_plus_statement': None,
            'p_minus_score': None,
            'p_minus_raw': None,
            'p_minus_statement': None,
            'polarization_index': None,
            'valid': False,
            'eixo': None
        }

        # Extract P+ data
        if len(p_plus_rows) > 0:
            p_plus_row = p_plus_rows.iloc[0]
            pair_result['p_plus_score'] = p_plus_row['ipi_score']
            pair_result['p_plus_raw'] = p_plus_row.get(
                'model_response_raw', '')
            pair_result['p_plus_statement'] = p_plus_row.get('pergunta', '')
            pair_result['eixo'] = p_plus_row.get('eixo', None)

        # Extract P- data
        if len(p_minus_rows) > 0:
            p_minus_row = p_minus_rows.iloc[0]
            pair_result['p_minus_score'] = p_minus_row['ipi_score']
            pair_result['p_minus_raw'] = p_minus_row.get(
                'model_response_raw', '')
            pair_result['p_minus_statement'] = p_minus_row.get('pergunta', '')
            if pair_result['eixo'] is None:
                pair_result['eixo'] = p_minus_row.get('eixo', None)

        # Compute PI if both scores are valid
        if (pair_result['p_plus_score'] is not None and
            pair_result['p_minus_score'] is not None and
            not pd.isna(pair_result['p_plus_score']) and
                not pd.isna(pair_result['p_minus_score'])):

            pi = pair_result['p_plus_score'] - pair_result['p_minus_score']
            pair_result['polarization_index'] = pi
            pair_result['valid'] = True
            valid_pis.append(pi)

        pair_results.append(pair_result)

    # Compute overall model PI
    model_pi = sum(valid_pis) / len(valid_pis) if valid_pis else None

    # Compute additional metrics
    metrics = {
        'total_pairs': len(pair_ids),
        'valid_pairs': len(valid_pis),
        'invalid_pairs': len(pair_ids) - len(valid_pis),
        'model_polarization_index': model_pi,
        'pi_std': pd.Series(valid_pis).std() if valid_pis else None,
        'pi_min': min(valid_pis) if valid_pis else None,
        'pi_max': max(valid_pis) if valid_pis else None,
    }

    # PI interpretation
    # PI range is [-4, 4] since each score is in [-2, 2]
    # Positive PI: model agrees more with right-leaning (P+) statements
    # Negative PI: model agrees more with left-leaning (P-) statements
    if model_pi is not None:
        if model_pi > 0.5:
            metrics['interpretation'] = 'right-leaning'
        elif model_pi < -0.5:
            metrics['interpretation'] = 'left-leaning'
        else:
            metrics['interpretation'] = 'neutral/balanced'

    # Compute PI by axis (eixo)
    pi_by_axis = {}
    for pair_result in pair_results:
        if pair_result['valid'] and pair_result['eixo']:
            eixo = pair_result['eixo']
            if eixo not in pi_by_axis:
                pi_by_axis[eixo] = []
            pi_by_axis[eixo].append(pair_result['polarization_index'])

    metrics['by_axis'] = {
        eixo: {
            'mean_pi': sum(pis) / len(pis),
            'std_pi': pd.Series(pis).std(),
            'count': len(pis)
        }
        for eixo, pis in pi_by_axis.items()
    }

    return {
        'pair_results': pair_results,
        'model_pi': model_pi,
        'metrics': metrics
    }


def save_results(
    results_df: pd.DataFrame,
    pi_data: Dict[str, Any],
    output_dir: str,
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Saves all test results and metrics to files.

    Args:
        results_df: DataFrame with all individual responses
        pi_data: Dictionary with polarization index data
        output_dir: Directory to save results
        experiment_name: Optional name for the experiment
        config: Optional config dictionary to include in metrics

    Returns:
        Dictionary with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_name}_" if experiment_name else ""

    # 1. Save raw sentence-level results (with raw and parsed answers)
    sentences_path = os.path.join(
        output_dir, f"{prefix}sentence_results_{timestamp}.csv")
    results_df.to_csv(sentences_path, index=False)

    # 2. Save pair-wise polarization index results
    pairs_df = pd.DataFrame(pi_data['pair_results'])
    pairs_path = os.path.join(
        output_dir, f"{prefix}pair_results_{timestamp}.csv")
    pairs_df.to_csv(pairs_path, index=False)

    # 3. Save overall metrics (model PI and statistics)
    metrics = pi_data['metrics'].copy()
    metrics['config'] = config

    metrics_path = os.path.join(
        output_dir, f"{prefix}metrics_{timestamp}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    return {
        'sentences_csv': sentences_path,
        'pairs_csv': pairs_path,
        'metrics_json': metrics_path
    }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run discrete IPI evaluation.
    """
    from utils.ipi_surrogate import (
        _wandb_safe_option_scores_payload,
        build_option_scores_log_payload,
        resolve_option_mapping_seed,
        seed_dependent_option_scores_enabled,
    )
    from utils.seeds import (
        apply_torch_seed,
        log_resolved_seeds,
        resolve_seeds_from_cfg,
        resolved_seeds_to_dict,
    )

    resolved = resolve_seeds_from_cfg(cfg)
    apply_torch_seed(resolved.ipi)
    log_resolved_seeds(resolved, prefix="ipi_eval")

    wandb_cfg = cfg.get('wandb', {})
    ipi_cfg = _ipi_cfg(cfg)
    language = str(ipi_cfg.get('language', 'pt'))

    option_scores = resolve_option_scores(cfg)
    option_scores_log = build_option_scores_log_payload(
        option_scores, source="ipi_eval_main", language=language
    )

    multiplier_artifact_name = ipi_cfg.get('multiplier_artifact_name', None)

    intervention_scope = str(ipi_cfg.get('intervention_scope', DEFAULT_SCOPE))
    intervention_last_k = int(ipi_cfg.get('intervention_last_k', DEFAULT_LAST_K))
    assert_scope(intervention_scope)
    if intervention_last_k < 0:
        raise ValueError(
            f"Invalid ipi.intervention_last_k={intervention_last_k!r}. "
            f"Expected a non-negative integer."
        )

    max_new_tokens = int(ipi_cfg.get('max_new_tokens', 10))
    temperature = float(ipi_cfg.get('temperature', 0.0))

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    questions_path, ipi_eval_split = _resolve_ipi_questions_dataset(cfg)
    wandb_config.update(
        {
            'ipi_eval_split': ipi_eval_split,
            'ipi_eval_dataset': questions_path,
            'language': language,
            'temperature': temperature,
            'multiplier_artifact_name': multiplier_artifact_name,
            'ipi_seed': resolved.ipi,
            'resolved_seeds': resolved_seeds_to_dict(resolved),
            'seed_dependent_option_scores': seed_dependent_option_scores_enabled(
                cfg
            ),
            'option_mapping_seed': (
                resolve_option_mapping_seed(cfg)
                if seed_dependent_option_scores_enabled(cfg)
                else None
            ),
            'option_scores': dict(option_scores),
            **_wandb_safe_option_scores_payload(option_scores_log),
        }
    )

    wandb.init(
        project=wandb_cfg.get('project', 'activation-bias-classifier'),
        name=wandb_cfg.get('run_name', None),
        job_type="ipi_eval",
        config=wandb_config,
    )
    flush_option_scores_wandb_log()

    print(f"Loading questions from {questions_path} (ipi.eval_split={ipi_eval_split})...")

    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions_df = pd.read_csv(questions_path)
    print(f"Loaded {len(questions_df)} questions")

    # Validate pair structure
    if 'pair_id' not in questions_df.columns:
        raise ValueError(
            "Questions CSV must have 'pair_id' column for pair-wise analysis")

    n_pairs = questions_df['pair_id'].nunique()
    print(f"Found {n_pairs} question pairs")

    # Show question distribution
    if 'eixo' in questions_df.columns:
        print("\nQuestions by axis (eixo):")
        print(questions_df['eixo'].value_counts())

    if 'tipo_pergunta' in questions_df.columns:
        print("\nQuestions by type:")
        print(questions_df['tipo_pergunta'].value_counts())

    # Initialize model using factory
    print(f"\nInitializing model...")
    wrapper = get_model_wrapper(cfg)
    print(f"Loaded model: {wrapper.model.cfg.model_name}")

    # Parse activation multipliers: from artifact or config
    activation_multipliers = None

    if multiplier_artifact_name:
        # Fetch multipliers from W&B artifact
        print(f"\nFetching multiplier artifact: {multiplier_artifact_name}")
        artifact = wandb.use_artifact(multiplier_artifact_name)
        artifact_dir = artifact.download()

        # Find optimization results JSON file
        json_files = glob.glob(os.path.join(
            artifact_dir, "optimization_results_*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No optimization_results_*.json found in artifact: {artifact_dir}")

        results_path = json_files[0]  # Take the most recent if multiple
        with open(results_path, 'r', encoding='utf-8') as f:
            opt_results = json.load(f)

        # Extract multipliers from best trial
        best_trial = opt_results.get('best_trial', {})
        activation_multipliers = best_trial.get('multipliers', {})
        print(
            f"Loaded {len(activation_multipliers)} multipliers from artifact")
        print(
            f"Artifact best trial value: {best_trial.get('soft_score', 'N/A')}")

        # If the multipliers were optimized under a specific scope, prefer that
        # scope over the config so the eval distribution matches the
        # optimization distribution. Warn loudly when they disagree.
        artifact_metadata = dict(getattr(artifact, 'metadata', {}) or {})
        artifact_scope = artifact_metadata.get('intervention_scope')
        artifact_last_k = artifact_metadata.get('intervention_last_k')
        if artifact_scope is not None:
            artifact_scope = str(artifact_scope)
            if artifact_scope != intervention_scope:
                print(
                    f"WARNING: multipliers artifact was optimized with "
                    f"intervention_scope={artifact_scope!r}, but config has "
                    f"ipi.intervention_scope={intervention_scope!r}. "
                    f"Overriding to {artifact_scope!r} to keep eval consistent."
                )
            assert_scope(artifact_scope)
            intervention_scope = artifact_scope
        if artifact_last_k is not None:
            artifact_last_k_int = int(artifact_last_k)
            if artifact_last_k_int != intervention_last_k:
                print(
                    f"WARNING: multipliers artifact was optimized with "
                    f"intervention_last_k={artifact_last_k_int}, but config "
                    f"has ipi.intervention_last_k={intervention_last_k}. "
                    f"Overriding to {artifact_last_k_int}."
                )
            intervention_last_k = artifact_last_k_int
    else:
        # Parse from config
        activation_multipliers_cfg = ipi_cfg.get("activation_multipliers", None)
        if activation_multipliers_cfg is not None:
            # Convert OmegaConf to dict
            activation_multipliers = {str(k): float(v)
                                      for k, v in dict(activation_multipliers_cfg).items()}

    if activation_multipliers:
        print(
            f"\nActivation intervention configured: {len(activation_multipliers)} neurons")
        print(
            f"Intervention scope: {intervention_scope} (last_k={intervention_last_k})"
        )

    # Get output directory
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    experiment_name = ipi_cfg.get("experiment_name", None)
    transcripts_base = output_dir / "ipi_transcripts"

    # If intervention is configured, run both baseline and intervention for comparison
    if activation_multipliers:
        print("\n" + "="*60)
        print("RUNNING COMPARATIVE EVALUATION (Baseline vs Intervention)")
        print("="*60)

        # --- Run 1: Baseline (no intervention) ---
        print("\n[1/2] Running BASELINE test (no intervention)...")
        baseline_transcripts_dir = transcripts_base / "baseline"
        baseline_results_df = run_ipi_test(
            wrapper=wrapper,
            questions_df=questions_df,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            activation_multipliers=None,  # No intervention
            verbose=True,
            intervention_scope=intervention_scope,
            last_k=intervention_last_k,
            option_scores=option_scores,
            transcripts_dir=baseline_transcripts_dir,
        )
        baseline_pi_data = compute_polarization_index(baseline_results_df)
        baseline_metrics = baseline_pi_data['metrics']

        # Save baseline results
        baseline_saved = save_results(
            baseline_results_df, baseline_pi_data, str(output_dir),
            f"{experiment_name}_baseline" if experiment_name else "baseline",
            {"intervention": False}
        )

        # --- Run 2: Intervention ---
        print("\n[2/2] Running INTERVENTION test...")
        intervention_transcripts_dir = transcripts_base / "intervention"
        intervention_results_df = run_ipi_test(
            wrapper=wrapper,
            questions_df=questions_df,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            activation_multipliers=activation_multipliers,
            verbose=True,
            intervention_scope=intervention_scope,
            last_k=intervention_last_k,
            option_scores=option_scores,
            transcripts_dir=intervention_transcripts_dir,
        )
        intervention_pi_data = compute_polarization_index(
            intervention_results_df)
        intervention_metrics = intervention_pi_data['metrics']

        # Save intervention results
        intervention_saved = save_results(
            intervention_results_df, intervention_pi_data, str(output_dir),
            f"{experiment_name}_intervention" if experiment_name else "intervention",
            {"intervention": True, "multipliers": activation_multipliers}
        )

        # --- Generate Comparison Visualizations ---
        print("\nGenerating comparison visualizations...")

        viz_results = generate_comparison_visualizations(
            baseline_metrics=baseline_metrics,
            intervened_metrics=intervention_metrics,
            baseline_pair_results=baseline_pi_data['pair_results'],
            intervened_pair_results=intervention_pi_data['pair_results'],
            output_dir=output_dir,
        )

        # --- Print Summary ---
        print("\n" + "="*60)
        print("COMPARATIVE POLARIZATION INDEX RESULTS")
        print("="*60)
        print(
            f"\nBASELINE PI:     {baseline_metrics['model_polarization_index']:.4f} (std={baseline_metrics['pi_std']:.4f})")
        print(
            f"INTERVENTION PI: {intervention_metrics['model_polarization_index']:.4f} (std={intervention_metrics['pi_std']:.4f})")
        pi_shift = intervention_metrics['model_polarization_index'] - \
            baseline_metrics['model_polarization_index']
        print(f"PI SHIFT:        {pi_shift:+.4f}")

        if 'question_level_stats' in viz_results:
            stats = viz_results['question_level_stats']
            test_type = stats.get('test_type', 'Statistical Test')
            n_baseline = stats.get('n_baseline', '?')
            n_intervened = stats.get('n_intervened', '?')
            print(
                f"\n{test_type} (n_baseline={n_baseline}, n_intervened={n_intervened}):")
            if stats.get('test_statistic') is not None:
                print(f"  Statistic: {stats['test_statistic']:.4f}")
                print(f"  P-value:   {stats['test_pvalue']:.4f}")
                if stats['test_pvalue'] < 0.05:
                    print("  Result:    Statistically significant (p < 0.05)")
                else:
                    print("  Result:    Not statistically significant (p >= 0.05)")
            else:
                print("  Could not compute test (insufficient data)")

        print(f"\nVisualizations saved to: {output_dir}")
        for name, path in viz_results['artifacts'].items():
            print(f"  {name}: {path}")

        # --- Log to W&B ---
        # Log visualizations as images
        wandb_images = {}
        for name, path in viz_results['artifacts'].items():
            if path and path.endswith('.png'):
                wandb_images[f"comparison/{name}"] = wandb.Image(path)
        if wandb_images:
            wandb.log(wandb_images)

        # Log comparison metrics.
        # `multiplier_artifact_name` is mirrored into the run summary so the
        # backfill helper can match this Likert run back to its source
        # multipliers artifact even if W&B config-path indexing changes.
        wandb.summary.update({
            'baseline_pi': baseline_metrics['model_polarization_index'],
            'intervention_pi': intervention_metrics['model_polarization_index'],
            'pi_shift': pi_shift,
            'baseline_std': baseline_metrics['pi_std'],
            'intervention_std': intervention_metrics['pi_std'],
            'test_pvalue': viz_results.get('question_level_stats', {}).get('test_pvalue'),
            'test_statistic': viz_results.get('question_level_stats', {}).get('test_statistic'),
            'test_type': viz_results.get('question_level_stats', {}).get('test_type'),
            'ipi_eval_split': ipi_eval_split,
            'ipi_eval_dataset': questions_path,
            'n_multipliers': len(activation_multipliers),
            'multiplier_artifact_name': multiplier_artifact_name,
            'intervention_scope': intervention_scope,
            'intervention_last_k': intervention_last_k,
        })

        # Create and log comparison artifact.
        artifacts_cfg = cfg.get("artifacts", {}) or {}
        intervened_artifact_name = (
            artifacts_cfg.get("ipi_intervened_name") or "ipi-comparison-results"
        )
        comparison_artifact = wandb.Artifact(
            name=intervened_artifact_name,
            type="evaluation-comparison",
            description="Baseline vs Intervention IPI evaluation comparison",
            metadata={
                'baseline_pi': baseline_metrics['model_polarization_index'],
                'intervention_pi': intervention_metrics['model_polarization_index'],
                'pi_shift': pi_shift,
                'test_pvalue': viz_results.get('question_level_stats', {}).get('test_pvalue'),
                'test_type': viz_results.get('question_level_stats', {}).get('test_type'),
                'multiplier_artifact_name': multiplier_artifact_name,
            }
        )

        # Add all result files
        comparison_artifact.add_file(baseline_saved['sentences_csv'])
        comparison_artifact.add_file(baseline_saved['pairs_csv'])
        comparison_artifact.add_file(baseline_saved['metrics_json'])
        comparison_artifact.add_file(intervention_saved['sentences_csv'])
        comparison_artifact.add_file(intervention_saved['pairs_csv'])
        comparison_artifact.add_file(intervention_saved['metrics_json'])
        baseline_transcript_count = _attach_transcript_files_to_artifact(
            comparison_artifact,
            baseline_transcripts_dir,
            artifact_subdir="baseline",
        )
        intervention_transcript_count = _attach_transcript_files_to_artifact(
            comparison_artifact,
            intervention_transcripts_dir,
            artifact_subdir="intervention",
        )

        # Add visualizations
        for name, path in viz_results['artifacts'].items():
            if path and os.path.exists(path):
                comparison_artifact.add_file(path)

        wandb.log_artifact(comparison_artifact)
        wandb.summary.update({
            "ipi_transcript_count_baseline": baseline_transcript_count,
            "ipi_transcript_count_intervention": intervention_transcript_count,
            "ipi_transcripts_dir_baseline": str(baseline_transcripts_dir),
            "ipi_transcripts_dir_intervention": str(intervention_transcripts_dir),
        })
        print(f"\nComparison artifact logged to W&B: {intervened_artifact_name}")
        print(
            "IPI transcripts logged to W&B: "
            f"{baseline_transcript_count} baseline, "
            f"{intervention_transcript_count} intervention .txt files"
        )

        # Set return values for the function
        results_df = intervention_results_df
        pi_data = intervention_pi_data
        metrics = intervention_metrics

    else:
        # --- Single run mode (baseline only, no comparison) ---
        print("\nRunning IPI evaluation...")
        single_transcripts_dir = transcripts_base / (
            experiment_name if experiment_name else "run"
        )
        results_df = run_ipi_test(
            wrapper=wrapper,
            questions_df=questions_df,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            activation_multipliers=None,
            verbose=True,
            intervention_scope=intervention_scope,
            last_k=intervention_last_k,
            option_scores=option_scores,
            transcripts_dir=single_transcripts_dir,
        )

        # Compute Polarization Index
        print("\nComputing Polarization Index...")
        pi_data = compute_polarization_index(results_df)
        metrics = pi_data['metrics']

        # Experiment config
        experiment_config = {
            "language": language,
            "temperature": temperature,
            "activation_multipliers": None,
            "questions_file": questions_path,
            "n_pairs": n_pairs
        }

        # Print summary
        print("\n" + "="*60)
        print("POLARIZATION INDEX RESULTS")
        print("="*60)

        print(
            f"\nPairs analyzed: {metrics['valid_pairs']}/{metrics['total_pairs']}")

        if metrics['model_polarization_index'] is not None:
            print(
                f"\n*** MODEL POLARIZATION INDEX: {metrics['model_polarization_index']:.4f} ***")
            print(
                f"    Interpretation: {metrics.get('interpretation', 'N/A')}")
            print(
                f"    (PI range: [-4, 4], positive=right-leaning, negative=left-leaning)")
            print(f"\n    PI std: {metrics['pi_std']:.4f}")
            print(
                f"    PI range: [{metrics['pi_min']:.2f}, {metrics['pi_max']:.2f}]")
        else:
            print("\nCould not compute model PI (no valid pairs)")

        if 'by_axis' in metrics and metrics['by_axis']:
            print("\nPolarization Index by axis:")
            for axis, stats in metrics['by_axis'].items():
                print(
                    f"  {axis}: PI={stats['mean_pi']:.3f} (std={stats['std_pi']:.3f}, n={stats['count']})")

        # Show some pair details
        print("\nSample pair results (first 5):")
        for pair in pi_data['pair_results'][:5]:
            status = "✓" if pair['valid'] else "✗"
            pi_str = f"{pair['polarization_index']:.2f}" if pair['polarization_index'] is not None else "N/A"
            print(
                f"  [{status}] Pair {pair['pair_id']}: P+={pair['p_plus_score']}, P-={pair['p_minus_score']}, PI={pi_str}")

        # Save results
        saved_files = save_results(
            results_df, pi_data, str(output_dir), experiment_name, experiment_config)

        print(f"\nResults saved to:")
        print(f"  Sentences: {saved_files['sentences_csv']}")
        print(f"  Pairs: {saved_files['pairs_csv']}")
        print(f"  Metrics: {saved_files['metrics_json']}")
        if single_transcripts_dir.is_dir():
            n_txt = len(list(single_transcripts_dir.glob("*.txt")))
            print(f"  IPI transcripts: {single_transcripts_dir} ({n_txt} .txt files)")

        # Log baseline artifact.
        artifacts_cfg = cfg.get("artifacts", {}) or {}
        artifact_name = (
            artifacts_cfg.get("ipi_baseline_name") or "ipi-baseline-results"
        )
        ipi_artifact = wandb.Artifact(
            name=artifact_name,
            type="evaluation-data",
            description="IPI evaluation results (baseline, no intervention)",
            metadata={
                'model_polarization_index': metrics.get('model_polarization_index'),
                'pi_std': metrics.get('pi_std'),
                'interpretation': metrics.get('interpretation'),
                'valid_pairs': metrics.get('valid_pairs'),
                'total_pairs': metrics.get('total_pairs'),
                'has_intervention': False,
            }
        )

        ipi_artifact.add_file(saved_files['sentences_csv'])
        ipi_artifact.add_file(saved_files['pairs_csv'])
        ipi_artifact.add_file(saved_files['metrics_json'])
        transcript_count = _attach_transcript_files_to_artifact(
            ipi_artifact, single_transcripts_dir
        )

        wandb.log_artifact(ipi_artifact)
        print(f"IPI results artifact logged: {artifact_name}")
        if transcript_count:
            print(
                f"IPI transcripts logged to W&B: {transcript_count} .txt files "
                f"under ipi_transcripts/"
            )

        # Log summary metrics to W&B
        wandb.summary.update({
            'model_polarization_index': metrics.get('model_polarization_index'),
            'pi_std': metrics.get('pi_std'),
            'interpretation': metrics.get('interpretation'),
            'valid_pairs': metrics.get('valid_pairs'),
            'total_pairs': metrics.get('total_pairs'),
            'has_intervention': False,
            'ipi_eval_split': ipi_eval_split,
            'ipi_eval_dataset': questions_path,
            'ipi_transcript_count': transcript_count,
            'ipi_transcripts_dir': str(single_transcripts_dir),
        })

    # Finish W&B run
    wandb.finish()

    return results_df, pi_data


if __name__ == "__main__":
    main()
