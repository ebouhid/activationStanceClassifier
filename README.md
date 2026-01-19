# Activation Stance Classifier

A tool for analyzing political stance in large language model activations using SVM classification.

## Project Overview

This project extracts neuron activation patterns from Llama 3.1 models and trains Support Vector Machine (SVM) classifiers to detect political Stance in text data. The analysis can be performed layer-wise or across all layers, enabling investigation of where stance representation emerges in different layers of the model.

## Project Structure

```
├── src/                          # Python source code
│   ├── extract_activations.py   # Extract model activations from text data
│   ├── train_eval_svc.py             # Train SVM classifier on activations
│   ├── llama_3dot1_wrapper.py   # Wrapper for Llama 3.1 model access
│   └── activation_df.py         # Activation data handling utilities
├── config/                       # Configuration files
│   ├── config.yaml              # Base configuration
│   └── allLayers_*.yaml         # Pre-configured setups for different layers/datasets
├── data/                         # Data and activation files
├── runs/                         # Experiment outputs and logs
├── wandb/                        # Weights & Biases experiment tracking logs
├── scripts/                      # Scripts used on the project
    ├── create_activation_datasets.sh  # Script to create activation datasets
    └── train_eval_SVCs.sh      # Run analysis
```

## Main Components

### 1. Activation Extraction (`extract_activations.py`)
- Loads text data with political stance labels
- Runs text through Llama 3.1 model
- Extracts neuron activations from specified layers
- Outputs activation data to CSV files

### 2. SVM Classifier (`train_eval_svc.py`)
- Loads activation data
- Performs feature selection (SelectKBest + MRMR)
- Trains SVM classifier with optional class balancing
- Generates classification reports, decision boundaries, and visualizations
- Integrates with Weights & Biases for experiment tracking

### 3. Model Wrapper (`llama_3dot1_wrapper.py`)
- Provides interface to Llama 3.1 model via HookedTransformer
- Handles activation retrieval through forward hooks
- Supports configurable model precision and device (CPU/CUDA)

## Configuration

Configuration is managed through YAML files and Hydra:

- **data.input_csv**: Path to input CSV with text and political labels
- **data.activations_file**: Output path for extracted activations
- **extraction.batch_size**: Batch size for model inference
- **extraction.layers**: Which model layers to extract (e.g., [19] for layer 19)
- **extraction.device**: "cuda" or "cpu"
- **extraction.max_length**: Maximum token sequence length
- **training.kernel**: SVM kernel type (e.g., "rbf")
- **feature_selection.n_features**: Number of features to select with MRMR
- **wandb.project**: Weights & Biases project name

## Usage
>Remember to set up your Weights & Biases account and login before running experiments.

First, ensure you have extracted activations using the `create_activation_datasets.sh` script.
```bash
./scripts/create_activation_datasets.sh
```
Then, run the SVM training and evaluation script:
```bash
./scripts/train_eval_SVCs.sh
```


## Data

The project expects input data with the following structure:
- **statement**: Text content to analyze
- **pol_label_human**: Political stance label (e.g., "left", "right")

## Experiment Tracking

Experiment results are tracked using Weights & Biases and saved in the `runs/` directory with timestamps, including:
- Classification metrics and reports
- Decision boundary visualizations
- Feature importance information
- Model hyperparameters and configuration

## Requirements

- Python 3.11+
- PyTorch with CUDA support (optional, CPU also supported)
- Hugging Face transformers and HookedTransformer
- scikit-learn for SVM and feature selection
- pandas, numpy for data handling
- Weights & Biases for experiment tracking
- Hydra for configuration management
- Check requirements.txt 😉

## Output

Analysis outputs are organized by timestamp in the `runs/` directory:
- Classification reports
- SVM decision boundary visualizations
- Feature selection results
- Complete run configurations
