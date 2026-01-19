import pandas as pd
import torch
import numpy as np
from typing import List, Optional, Union


class ActivationDataFrame:
    """
    Efficiently accumulates activation vectors and converts them to a pandas DataFrame.
    """

    def __init__(self, layers: List[int], d_model: int):
        """
        Args:
            layers: List of layer indices used for extraction (sorted).
            d_model: Dimension of activations per layer.
        """
        # Store data in lists first to avoid expensive DataFrame resizing operations
        self._activations_list: List[np.ndarray] = []
        self._labels_list: List[str] = []
        self._df: Optional[pd.DataFrame] = None
        self._layers = sorted(layers)
        self._d_model = d_model

    def add_batch(self, activations: torch.Tensor, labels: List[str]):
        """
        Adds a batch of activations and labels.

        Args:
            activations (torch.Tensor): Shape [batch_size, d_model].
                IMPORTANT: The wrapper returns [batch, seq_len, d_model].
                You must select a specific token (e.g., the last token) before passing it here.
            labels (List[str]): List of class labels for the batch.
        """
        if activations.ndim != 2:
            raise ValueError(f"Expected activations shape [batch, d_model], got {activations.shape}. "
                             "Did you forget to select a specific token (e.g., [:, -1, :])?")

        if activations.shape[0] != len(labels):
            raise ValueError(
                f"Batch size mismatch: {activations.shape[0]} activations vs {len(labels)} labels")

        # Detach from graph, move to CPU, and convert to numpy float32
        # float32 is standard for pandas; float16 saves memory but can be tricky with some pandas backends
        acts_np = activations.detach().cpu().to(dtype=torch.float32).numpy()

        self._activations_list.append(acts_np)
        self._labels_list.extend(labels)
        self._df = None  # Invalidate cached DataFrame

    def get_df(self) -> pd.DataFrame:
        """
        Constructs and returns the pandas DataFrame.
        """
        if self._df is not None:
            return self._df

        if not self._activations_list:
            return pd.DataFrame()

        # 1. Concatenate all numpy arrays (Much faster than creating DF row by row)
        X = np.concatenate(self._activations_list, axis=0)

        # 2. Create column names (layer_m-neuron_n)
        feature_cols = []
        for layer in self._layers:
            for neuron in range(self._d_model):
                feature_cols.append(f"layer_{layer}-neuron_{neuron}")

        # 3. Create DataFrame
        self._df = pd.DataFrame(X, columns=feature_cols)

        # 4. Add class column
        self._df["class"] = self._labels_list

        return self._df

    def save(self, filepath: str):
        """
        Saves the DataFrame. Strongly recommend using .parquet extension.
        """
        df = self.get_df()
        if filepath.endswith(".parquet"):
            df.to_parquet(filepath, index=False)
        else:
            # CSV will be very slow for 4000+ columns
            df.to_csv(filepath, index=False)
