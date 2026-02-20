# Activation Stance Classifier

A research framework for **analyzing and steering political stance** in Large Language Model (LLM) activations without fine-tuning. This project implements an end-to-end pipeline from activation extraction to bias intervention and evaluation.

## Research Overview

This project investigates how political bias manifests in LLM internal representations and develops methods to **steer model behavior** through targeted activation interventions. The approach:

1. **Extracts** neuron activations from politically-labeled text
2. **Identifies** politically-relevant neurons using SVM feature selection
3. **Optimizes** intervention multipliers to shift model political stance
4. **Evaluates** the intervention's effect using Likert-scale questionnaires
5. **Validates** that interventions don't harm general capabilities (PoETa benchmark)

## Pipeline Architecture
⚠️ **Note**: PoETa evaluation is currently not fully implemented yet. ⚠️
The project uses **W&B Artifacts** for full reproducibility and lineage tracking:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          W&B ARTIFACT PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. EXTRACTION                                                              │
│     extract_activations.py                                                  │
│     └──► activations-{dataset}-{model}-{layers}:latest                      │
│                            │                                                │
│                            ▼                                                │
│  2. FEATURE SELECTION                                                       │
│     train_eval_svc.py                                                       │
│     └──► svm-feature-ranking:latest                                         │
│                            │                                                │
│                            ▼                                                │
│  3. OPTIMIZATION                                                            │
│     optimize_intervention.py                                                │
│     └──► intervention-multipliers:latest                                    │
│                            │                                                │
│              ┌─────────────┴─────────────┐                                  │
│              ▼                           ▼                                  │
│  4. EVALUATION                                                              │
│     likert_scale_test.py              likert_scale_test.py                  │
│     (no intervention)                 (with intervention)                   │
│     └──► likert-baseline-results      └──► likert-intervened-results        │
│              │                           │                                  │
│              └───────────┬───────────────┘                                  │
│                          ▼                                                  │
│  5. VISUALIZATION                                                           │
│     plot_pi_shift.py                                                        │
│     └──► W&B: radar, fluidity, boxplot, parallel plots                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── src/                              # Core Python modules
│   ├── extract_activations.py       # Step 1: Extract model activations
│   ├── train_eval_svc.py            # Step 2: Train SVM, select features
│   ├── optimize_intervention.py     # Step 3: Optuna optimization
│   ├── likert_scale_test.py         # Step 4: Polarization Index evaluation
│   ├── compile_target_neurons.py    # Helper: Select neurons from ranking
│   ├── model_factory.py             # Model loading (Llama/Gemma)
│   ├── llama_3dot1_wrapper.py       # Llama 3.1 wrapper
│   ├── gemma_3_wrapper.py           # Gemma 3 wrapper
│   ├── activation_df.py             # Activation data utilities
│   └── poeta_evaluator.py           # PoETa benchmark evaluation
├── visualizations/
│   └── plot_pi_shift.py             # Step 5: Generate comparison plots
├── config/                           # Hydra configuration files
│   ├── config.yaml                  # Base defaults
│   ├── optimization_llama.yaml      # Llama optimization config
│   ├── optimization_gemma.yaml      # Gemma optimization config
│   ├── likert_eval.yaml             # Likert evaluation config
│   ├── poeta_eval.yaml              # PoETa benchmark config
│   └── poeta_eval_compare.yaml      # PoETa comparison config
├── likert/
│   └── questions_anderson.csv       # Likert questionnaire (paired P+/P-)
├── data/                             # Activation datasets
├── runs/                             # Experiment outputs
└── wandb/                            # W&B tracking logs
```

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

### Full Pipeline Execution

```bash
# 1. Extract activations from your dataset
python src/extract_activations.py \
  data.input_csv="/path/to/political_texts.csv" \
  extraction.layers="all"

# 2. Train SVM and identify politically-relevant neurons
python src/train_eval_svc.py \
  data.activations_artifact_name="activations-political_texts-llama-all:latest"

# 3. Optimize intervention multipliers
python src/optimize_intervention.py --config-name optimization_llama \
  optimization.feature_artifact_name="svm-feature-ranking:latest" \
  optimization.direction="minimize"  # Shift LEFT

# 4a. Run baseline evaluation (no intervention)
python src/likert_scale_test.py --config-name likert_eval

# 4b. Run intervened evaluation
python src/likert_scale_test.py --config-name likert_eval \
  likert.multiplier_artifact_name="intervention-multipliers:latest"

# 5. Generate comparison visualizations
python visualizations/plot_pi_shift.py \
  --baseline-artifact "likert-baseline-results:latest" \
  --intervened-artifact "likert-intervened-results:latest"
```

## Configuration

All scripts use [Hydra](https://hydra.cc/) for configuration management. Override any parameter via CLI:

```bash
# Change model
python src/script.py model.name="google/gemma-3-4b-it" model.wrapper="gemma"

# Change dataset
python src/script.py data.input_csv="/path/to/different_data.csv"

# Change optimization direction
python src/optimize_intervention.py optimization.direction="maximize"  # Shift RIGHT
```

### Config Files

| File | Purpose |
|------|---------|
| `config.yaml` | Base defaults for extraction and SVM training |
| `optimization_llama.yaml` | Llama neuron intervention optimization |
| `optimization_gemma.yaml` | Gemma neuron intervention optimization |
| `likert_eval.yaml` | Likert scale political stance evaluation |
| `poeta_eval.yaml` | PoETa benchmark (capability validation) |

## Key Concepts

### Polarization Index (PI)

The **Polarization Index** measures model political stance on a scale of **[-4, +4]**:
- **Positive PI**: Model agrees more with right-leaning statements
- **Negative PI**: Model agrees more with left-leaning statements
- **Zero PI**: Neutral/balanced responses

Computed from paired questions (P+ right-leaning, P- left-leaning):
```
PI_pair = score(P+) - score(P-)
Model_PI = mean(all pair PIs)
```

### Intervention Multipliers

Identified neurons have their activations scaled during generation:
```
activation_new = activation_original × multiplier
```

Multipliers are optimized via Optuna to shift the model's PI in the desired direction while minimizing deviation from baseline behavior.

### W&B Artifacts

Every pipeline stage produces versioned artifacts:
- **activations-{dataset}-{model}-{layers}**: Extracted activation CSVs
- **svm-feature-ranking**: Neuron importance rankings
- **intervention-multipliers**: Optimized multiplier JSON
- **likert-baseline-results** / **likert-intervened-results**: Evaluation CSVs and metrics

Use `:latest` to always fetch the most recent version, or pin specific versions for reproducibility.

## Experiment Tracking

All experiments are tracked in Weights & Biases:
- **Metrics**: Balanced accuracy, PI scores, Wilcoxon statistics
- **Tables**: Feature rankings, classification reports
- **Images**: Decision boundaries, radar charts, PI shift plots
- **Artifacts**: Full data lineage from extraction to evaluation

## Supported Models

| Model | Wrapper | Config |
|-------|---------|--------|
| meta-llama/Llama-3.1-8B-Instruct | `llama` | Default |
| google/gemma-3-4b-it | `gemma` | `model.wrapper="gemma"` |

## Common Workflows

### Compare Llama vs Gemma interventions

```bash
# Llama pipeline
python src/optimize_intervention.py --config-name optimization_llama
python src/likert_scale_test.py --config-name likert_eval \
  likert.multiplier_artifact_name="intervention-multipliers:v1"

# Gemma pipeline
python src/optimize_intervention.py --config-name optimization_gemma
python src/likert_scale_test.py --config-name likert_eval \
  model.name="google/gemma-3-4b-it" model.wrapper="gemma" \
  likert.multiplier_artifact_name="intervention-multipliers:v2"
```

### Validate intervention doesn't harm capabilities

```bash
python src/poeta_evaluator.py --config-name poeta_eval_compare \
  multiplier_artifact_name="intervention-multipliers:latest"
```

### Resume interrupted optimization

```bash
python src/optimize_intervention.py --config-name optimization_llama \
  optimization.storage="sqlite:///runs/optuna_persist/study.db" \
  optimization.load_if_exists=true
```

## Output Examples

### Feature Ranking (from SVM)
```
rank  feature              selection_count  selection_frequency
1     layer_15-neuron_2058  3               1.0
2     layer_20-neuron_2212  3               1.0
3     layer_16-neuron_122   3               1.0
...
```

### PI Shift Results
```
Baseline PI:    +0.847 (right-leaning)
Intervened PI:  +0.123 (near neutral)
Shift:          -0.724
Wilcoxon p:     0.0012 (significant)
```

## Requirements

- Python 3.11+
- PyTorch with CUDA support (recommended)
- 16GB+ GPU memory (I used an RTX 3090)
- Weights & Biases account

See `requirements.txt` for full dependencies.
