import pandas as pd
import torch
from tqdm import tqdm
from llama_3dot1_wrapper import Llama3dot1Wrapper
from activation_df import ActivationDataFrame
import os
import hydra
from omegaconf import DictConfig


def get_last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Finds the index of the last non-padding token for each sequence in the batch.
    Assuming attention_mask is 1 for tokens and 0 for padding.
    """
    return attention_mask.sum(dim=1) - 1


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Configuration from Hydra
    input_path = hydra.utils.to_absolute_path(cfg.data.input_csv)
    output_path = cfg.data.activations_file
    batch_size = cfg.extraction.batch_size
    device = cfg.extraction.device if torch.cuda.is_available(
    ) and cfg.extraction.device == "cuda" else "cpu"

    # 1. Load Data
    print(f"Loading data from {input_path}...")
    # For demonstration, creating a dummy dataframe if file doesn't exist
    if not os.path.exists(input_path):
        print("Input file not found. Creating dummy data for demonstration.")
        df = pd.DataFrame({
            'statement': ["This is a test sentence.", "Another political statement.", "Short one."] * 10,
            'pol_label_human': ["neutral", "political", "neutral"] * 10
        })
    else:
        df = pd.read_csv(input_path)
        # Ensure columns exist
        if 'statement' not in df.columns or 'pol_label_human' not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'statement' and 'pol_label_human' columns.")

    print(f"Loaded {len(df)} samples.")

    # 2. Initialize Model
    print(f"Initializing model on {device}...")
    wrapper = Llama3dot1Wrapper(device=device)

    # Ensure tokenizer has a pad token
    if wrapper.model.tokenizer.pad_token is None:
        wrapper.model.tokenizer.pad_token = wrapper.model.tokenizer.eos_token

    # Resolve layers list (handle 'all' option)
    layers_cfg = cfg.extraction.layers
    if isinstance(layers_cfg, str) and layers_cfg.lower() == 'all':
        layers = list(range(wrapper.n_layers))
    else:
        layers = list(layers_cfg)

    # 3. Initialize Accumulator with layer info
    d_model = wrapper.model.cfg.d_model
    activation_df = ActivationDataFrame(layers=layers, d_model=d_model)

    # 4. Processing Loop
    print("Starting extraction loop...")

    # Create batches
    total_samples = len(df)

    for start_idx in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_df = df.iloc[start_idx:end_idx]

        texts = batch_df['statement'].tolist()
        labels = batch_df['pol_label_human'].tolist()

        # Tokenize
        encoding = wrapper.model.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=cfg.extraction.max_length
        ).to(device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        try:
            layer_activations = wrapper.get_layer_activations(
                input_ids, layers=cfg.extraction.layers).to(device)
        except Exception as e:
            print(f"Error processing batch {start_idx}-{end_idx}: {e}")
            continue

        padding_side = wrapper.model.tokenizer.padding_side

        batch_indices = torch.arange(input_ids.shape[0], device=device)

        if padding_side == 'left':
            last_token_indices = -1
            final_activations = layer_activations[:, -1, :]
        else:
            last_token_indices = attention_mask.sum(dim=1).to(device) - 1
            final_activations = layer_activations[batch_indices,
                                                  last_token_indices, :]

        # Add to accumulator
        activation_df.add_batch(final_activations, labels)

    # 5. Save Results
    print(f"Saving results to {output_path}...")
    # Hydra changes cwd to the run dir, so we just save to the filename
    activation_df.save(output_path)
    print(f"Done. Saved to {os.getcwd()}/{output_path}")


if __name__ == "__main__":
    main()
