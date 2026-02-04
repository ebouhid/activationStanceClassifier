"""
PoETa Benchmark Evaluator for Llama 3.1 with Activation Interventions

This module provides integration with the PoETa V2 benchmark for evaluating
baseline and intervened models using the Llama3dot1Wrapper.

Usage:
    # From command line with Hydra config:
    python src/poeta_evaluator.py
    
    # Or programmatically:
    from poeta_evaluator import run_poeta_evaluation
    results = run_poeta_evaluation(
        activation_multipliers={'layer_10-neuron_512': 0.5},
        tasks=['assin_rte_greedy', 'enem_greedy'],
        output_path='results/poeta_intervened.json'
    )
"""

from llama_3dot1_wrapper import Llama3dot1Wrapper
import os
import sys
import json
import torch
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Iterable
from pathlib import Path
from contextlib import contextmanager

import hydra
from omegaconf import DictConfig, OmegaConf


class TeeLogger:
    """Logger that writes to both console and file simultaneously."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.file = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


@contextmanager
def tee_output(log_file: Path):
    """Context manager to tee stdout/stderr to a log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    tee_stdout = TeeLogger(log_file)
    tee_stderr = TeeLogger(log_file.with_suffix('.err.log'))

    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    try:
        yield log_file
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee_stdout.close()
        tee_stderr.close()


# Add PoETa to path (sibling directory) - must be added before lm_eval imports
# This ensures we use the local fork with fixes applied
POETA_PATH = Path(__file__).parent.parent.parent / "PoETaV2"
PROJECT_PATH = Path(__file__).parent.parent

if POETA_PATH.exists():
    sys.path.insert(0, str(POETA_PATH))

# Change to PoETa directory for conversation.py import
_original_cwd = os.getcwd()
os.chdir(POETA_PATH)

try:
    from lm_eval.base import BaseLM
    from lm_eval import tasks, evaluator
    from lm_eval.utils import stop_sequences_criteria
finally:
    os.chdir(_original_cwd)


class IntervenedLlamaLM(BaseLM):
    """
    PoETa-compatible Language Model wrapper for Llama 3.1 with activation interventions.

    This class bridges the Llama3dot1Wrapper (which uses TransformerLens/HookedTransformer)
    with the PoETa evaluation framework, enabling evaluation of models with modified
    internal activations.
    """

    def __init__(
        self,
        device: str = "cuda",
        pretrained: str = "meta-llama/Llama-3.1-8B-Instruct",
        batch_size: int = 1,
        activation_multipliers: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the intervened Llama model for PoETa evaluation.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            pretrained: HuggingFace model identifier
            batch_size: Batch size for evaluation
            activation_multipliers: Dict mapping 'layer_X-neuron_Y' to multiplier values
                                   for activation interventions. None for baseline.
        """
        super().__init__()

        self._device = torch.device(device)
        self.batch_size_per_gpu = batch_size
        self.activation_multipliers = activation_multipliers or {}

        # Initialize our custom wrapper
        print(f"Loading model: {pretrained}")
        print(
            f"Activation interventions: {len(self.activation_multipliers)} neurons")
        self.wrapper = Llama3dot1Wrapper(model_name=pretrained, device=device)

        # Get references to model and tokenizer from wrapper
        self.model = self.wrapper.model
        self.tokenizer = self.model.tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        # Parse intervention layers for hook setup
        self._intervention_layers = self._parse_intervention_layers()

    def _parse_intervention_layers(self) -> Dict[int, Dict[int, float]]:
        """Parse activation_multipliers into per-layer neuron dictionaries."""
        layer_neuron_multipliers = {}
        for feature_name, multiplier in self.activation_multipliers.items():
            parts = feature_name.split('-')
            layer_idx = int(parts[0].split('_')[1])
            neuron_idx = int(parts[1].split('_')[1])
            if layer_idx not in layer_neuron_multipliers:
                layer_neuron_multipliers[layer_idx] = {}
            layer_neuron_multipliers[layer_idx][neuron_idx] = multiplier
        return layer_neuron_multipliers

    def _make_intervention_hook(self, neuron_multipliers: Dict[int, float]):
        """Create a hook function that applies neuron-specific multipliers."""
        def hook(resid_pre: torch.Tensor, hook):
            modified = resid_pre.clone()
            for neuron_idx, multiplier in neuron_multipliers.items():
                modified[:, :, neuron_idx] = modified[:,
                                                      :, neuron_idx] * multiplier
            return modified
        return hook

    def _get_intervention_hooks(self) -> List[tuple]:
        """Build list of (hook_point, hook_fn) tuples for interventions."""
        fwd_hooks = []
        for layer_idx, neuron_mults in self._intervention_layers.items():
            hook_point = f"blocks.{layer_idx}.hook_resid_pre"
            fwd_hooks.append(
                (hook_point, self._make_intervention_hook(neuron_mults)))
        return fwd_hooks

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        """Maximum context length the model can handle."""
        try:
            # Prefer tokenizer's model_max_length as it's more accurate for Llama 3.1
            if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length < 1e10:
                return self.tokenizer.model_max_length
            # Fallback to TransformerLens cfg.n_ctx
            if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'n_ctx'):
                return self.model.cfg.n_ctx
            return 131072  # Llama 3.1 default context
        except Exception:
            return 131072

    @property
    def max_gen_toks(self):
        return 2048

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional activation interventions.

        Args:
            inps: Input token IDs [batch, sequence]

        Returns:
            Logits tensor [batch, sequence, vocab_size]
        """
        with torch.no_grad():
            if self._intervention_layers:
                # Run with intervention hooks
                fwd_hooks = self._get_intervention_hooks()
                logits = self.model.run_with_hooks(
                    inps.to(self._device),
                    fwd_hooks=fwd_hooks
                )
            else:
                # Standard forward pass (baseline)
                logits = self.model(inps.to(self._device))

        return logits[:, :, :self.vocab_size]

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop_sequences: List[str]
    ) -> torch.Tensor:
        """
        Generate text with optional activation interventions.

        Args:
            context: Input token IDs [batch, seq_len]
            max_length: Maximum total length (input + generated)
            stop_sequences: List of strings to stop generation at

        Returns:
            Generated token IDs including input
        """
        max_new_tokens = max_length - context.shape[1]

        # Build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop_sequences, context.shape[1], context.shape[0]
        )

        with torch.no_grad():
            if self._intervention_layers:
                # Add hooks temporarily for generation
                fwd_hooks = self._get_intervention_hooks()
                for hook_point, hook_fn in fwd_hooks:
                    self.model.add_hook(hook_point, hook_fn)

                try:
                    output = self.model.generate(
                        context.to(self._device),
                        max_new_tokens=max_new_tokens,
                        stop_at_eos=True,
                        eos_token_id=self.eot_token_id,
                        do_sample=False,
                        verbose=False
                    )
                finally:
                    self.model.reset_hooks()
            else:
                # Standard generation (baseline)
                output = self.model.generate(
                    context.to(self._device),
                    max_new_tokens=max_new_tokens,
                    stop_at_eos=True,
                    eos_token_id=self.eot_token_id,
                    do_sample=False,
                    verbose=False
                )

        return output


def run_poeta_evaluation(
    activation_multipliers: Optional[Dict[str, float]] = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    tasks_list: Optional[List[str]] = None,
    num_fewshot: int = 0,
    prompt_modes: str = "dynamic-random",
    batch_size: int = 1,
    device: str = "cuda",
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    description_dict_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run PoETa V2 benchmark evaluation on a Llama model with optional interventions.

    Args:
        activation_multipliers: Dict of 'layer_X-neuron_Y' -> multiplier for interventions.
                               Use None or {} for baseline evaluation.
        model_name: HuggingFace model identifier
        tasks_list: List of PoETa task names to run. None runs all tasks.
        num_fewshot: Number of few-shot examples
        prompt_modes: Prompt mode(s), comma-separated
        batch_size: Evaluation batch size
        device: Device to run on ('cuda' or 'cpu')
        limit: Limit number of examples per task (for testing)
        output_path: Path to save results JSON
        description_dict_path: Path to task description JSON

    Returns:
        Dictionary containing evaluation results
    """
    # Default to all tasks if none specified
    if tasks_list is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = tasks_list

    # Parse prompt modes
    prompt_modes_list = prompt_modes.split(",")

    # Load description dict if provided
    description_dict = {}
    if description_dict_path:
        desc_path = Path(description_dict_path)
        if not desc_path.is_absolute():
            desc_path = POETA_PATH / description_dict_path
        if desc_path.exists():
            with open(desc_path, 'r') as f:
                description_dict = json.load(f)

    # Create our custom model
    model = IntervenedLlamaLM(
        device=device,
        pretrained=model_name,
        batch_size=batch_size,
        activation_multipliers=activation_multipliers,
    )

    # Determine output directory (convert to absolute path for saving outside PoETa dir)
    output_dir = None
    abs_output_path = None
    if output_path:
        abs_output_path = Path(output_path)
        if not abs_output_path.is_absolute():
            abs_output_path = PROJECT_PATH / output_path
        abs_output_path = abs_output_path.resolve()
        output_dir = str(abs_output_path.parent)
        if output_dir == "":
            output_dir = "."

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Running PoETa V2 Evaluation")
    print(f"Model: {model_name}")
    print(f"Interventions: {len(activation_multipliers or {})} neurons")
    print(f"Tasks: {len(task_names)} tasks")
    print(f"Limit: {limit}")
    print(f"{'='*60}\n")

    # Change to PoETa directory for task data loading
    original_cwd = os.getcwd()
    os.chdir(POETA_PATH)

    try:
        results = evaluator.simple_evaluate(
            model=model,
            model_args="",  # Not used since we pass model instance
            tasks=task_names,
            num_fewshot=num_fewshot,
            prompt_modes=prompt_modes_list,
            batch_size=batch_size,
            device=device,
            no_cache=True,
            limit=limit,
            description_dict=description_dict,
            conversation_template=None,
            prompt_as_single_user_message=False,
            check_integrity=False,
            output_dir=output_dir,
        )
    finally:
        os.chdir(original_cwd)

    # Add metadata to results
    results['metadata'] = {
        'model_name': model_name,
        'activation_multipliers': activation_multipliers,
        'num_interventions': len(activation_multipliers or {}),
        'evaluation_type': 'intervened' if activation_multipliers else 'baseline',
        'timestamp': datetime.now().isoformat(),
        'tasks': task_names,
        'num_fewshot': num_fewshot,
        'limit': limit,
    }

    def make_serializable(obj):
        """Recursively convert non-serializable objects to strings."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert non-serializable objects to their string representation
            return str(obj)

    serializable_results = make_serializable(results)

    # Save results
    if abs_output_path:
        abs_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(abs_output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {abs_output_path}")

    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(evaluator.make_table(results))

    return serializable_results


def compare_baseline_vs_intervened(
    activation_multipliers: Dict[str, float],
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    tasks_list: Optional[List[str]] = None,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    output_dir: str = "runs",
    save_logs: bool = True,
) -> Dict[str, Any]:
    """
    Run PoETa evaluation on both baseline and intervened models, then compare.

    Args:
        activation_multipliers: Interventions to apply to the model
        model_name: HuggingFace model identifier
        tasks_list: List of tasks to evaluate
        num_fewshot: Few-shot examples
        limit: Example limit per task
        output_dir: Base directory to save results (will create runs/date/poeta_time/)
        save_logs: Whether to save terminal output to log files

    Returns:
        Dictionary with 'baseline', 'intervened', and 'comparison' results
    """
    # Follow convention: runs/date/poeta_time/
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    run_dir = Path(output_dir) / date_str / f"poeta_{time_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file
    log_file = run_dir / "evaluation.log"

    def run_evaluations():
        # Run baseline
        print("\n" + "#"*60)
        print("# BASELINE EVALUATION")
        print("#"*60)
        baseline_results = run_poeta_evaluation(
            activation_multipliers=None,
            model_name=model_name,
            tasks_list=tasks_list,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=str(run_dir / "baseline_results.json"),
            description_dict_path="description.json",
        )

        # Run intervened
        print("\n" + "#"*60)
        print("# INTERVENED EVALUATION")
        print("#"*60)
        intervened_results = run_poeta_evaluation(
            activation_multipliers=activation_multipliers,
            model_name=model_name,
            tasks_list=tasks_list,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=str(run_dir / "intervened_results.json"),
            description_dict_path="description.json",
        )

        # Compute comparison
        comparison = compute_comparison(baseline_results, intervened_results)

        # Save comparison
        comparison_path = run_dir / "comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print("\n" + "="*60)
        print("COMPARISON: Intervened vs Baseline")
        print("="*60)
        for task, metrics in comparison.get('per_task', {}).items():
            print(f"\n{task}:")
            for metric, delta in metrics.items():
                sign = "+" if delta > 0 else ""
                print(f"  {metric}: {sign}{delta:.4f}")

        if 'overall' in comparison:
            print(f"\nOverall change: {comparison['overall']:+.4f}")

        return baseline_results, intervened_results, comparison

    # Run with or without logging
    if save_logs:
        with tee_output(log_file):
            print(f"Log file: {log_file}")
            print(f"Run directory: {run_dir}")
            print(f"Timestamp: {datetime.now().isoformat()}")
            print("\n" + "="*60)
            baseline_results, intervened_results, comparison = run_evaluations()
    else:
        baseline_results, intervened_results, comparison = run_evaluations()

    # Save config used for this run
    config_path = run_dir / "config.json"
    config_data = {
        'model_name': model_name,
        'activation_multipliers': activation_multipliers,
        'tasks': tasks_list,
        'num_fewshot': num_fewshot,
        'limit': limit,
        'timestamp': datetime.now().isoformat(),
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    return {
        'baseline': baseline_results,
        'intervened': intervened_results,
        'comparison': comparison,
        'output_dir': str(run_dir),
        'log_file': str(log_file) if save_logs else None,
    }


def compute_comparison(
    baseline: Dict[str, Any],
    intervened: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute metric differences between baseline and intervened results."""
    comparison = {'per_task': {}}

    baseline_results = baseline.get('results', {})
    intervened_results = intervened.get('results', {})

    deltas = []
    for task in baseline_results:
        if task in intervened_results:
            comparison['per_task'][task] = {}
            for metric in baseline_results[task]:
                if metric in intervened_results[task]:
                    base_val = baseline_results[task][metric]
                    int_val = intervened_results[task][metric]
                    if isinstance(base_val, (int, float)) and isinstance(int_val, (int, float)):
                        delta = int_val - base_val
                        comparison['per_task'][task][metric] = delta
                        deltas.append(delta)

    if deltas:
        comparison['overall'] = sum(deltas) / len(deltas)

    return comparison


# Hydra configuration for running from command line
@hydra.main(config_path="../config", config_name="poeta_eval", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for Hydra-based PoETa evaluation.

    Config should contain:
        - model_name: HuggingFace model path
        - tasks: List of task names or 'all'
        - num_fewshot: Number of few-shot examples
        - limit: Example limit (null for full evaluation)
        - activation_multipliers: Dict of interventions (null for baseline)
        - compare_baseline: Whether to run comparison mode
        - output_dir: Where to save results
    """
    # Change back to project directory (Hydra changes cwd to outputs/)
    os.chdir(PROJECT_PATH)

    print(OmegaConf.to_yaml(cfg))

    # Convert OmegaConf to dict for activation_multipliers
    activation_multipliers = None
    if cfg.get('activation_multipliers'):
        activation_multipliers = OmegaConf.to_container(
            cfg.activation_multipliers)

    # Parse tasks
    tasks_list = None
    if cfg.get('tasks') and cfg.tasks != 'all':
        if isinstance(cfg.tasks, str):
            tasks_list = cfg.tasks.split(',')
        else:
            # It's already a list (ListConfig)
            tasks_list = list(cfg.tasks)

    if cfg.get('compare_baseline', False):
        # Run comparison mode
        results = compare_baseline_vs_intervened(
            activation_multipliers=activation_multipliers or {},
            model_name=cfg.get(
                'model_name', 'meta-llama/Llama-3.1-8B-Instruct'),
            tasks_list=tasks_list,
            num_fewshot=cfg.get('num_fewshot', 0),
            limit=cfg.get('limit'),
            output_dir=cfg.get('output_dir', 'runs'),
            save_logs=cfg.get('save_logs', True),
        )
    else:
        # Run single evaluation - follow convention: runs/date/poeta_time/
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        eval_type = 'intervened' if activation_multipliers else 'baseline'
        run_dir = Path(cfg.get('output_dir', 'runs')) / \
            date_str / f"poeta_{time_str}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(run_dir / f"{eval_type}_results.json")
        log_file = run_dir / "evaluation.log"

        def run_single_eval():
            return run_poeta_evaluation(
                activation_multipliers=activation_multipliers,
                model_name=cfg.get(
                    'model_name', 'meta-llama/Llama-3.1-8B-Instruct'),
                tasks_list=tasks_list,
                num_fewshot=cfg.get('num_fewshot', 0),
                limit=cfg.get('limit'),
                output_path=output_path,
                description_dict_path=cfg.get(
                    'description_dict_path', 'description.json'),
            )

        if cfg.get('save_logs', True):
            with tee_output(log_file):
                print(f"Log file: {log_file}")
                print(f"Run directory: {run_dir}")
                print(f"Timestamp: {datetime.now().isoformat()}")
                print("\n" + "="*60)
                results = run_single_eval()
        else:
            results = run_single_eval()

        # Save config used for this run
        config_path = run_dir / "config.json"
        config_data = {
            'model_name': cfg.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct'),
            'activation_multipliers': activation_multipliers,
            'tasks': tasks_list,
            'num_fewshot': cfg.get('num_fewshot', 0),
            'limit': cfg.get('limit'),
            'timestamp': datetime.now().isoformat(),
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    return results


if __name__ == "__main__":
    main()
