import torch
from transformer_lens import HookedTransformer
from typing import List, Union


class Llama3dot1Wrapper:
    """
    A wrapper to load a HookedTransformer model and retrieve layer activations 
    using forward hooks.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda"):
        """
        Initializes the model, mimicking the configuration in the provided files.

        Args:
            model_name (str): The name of the model to load.
            device (str): The device (e.g., 'cuda', 'cpu') to load the model onto.
        """
        self.device = device

        # Load the base model using HookedTransformer (FP16 as seen in model_utils.py)
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=torch.float16
        )

        # Ensure a pad token exists for masking if needed
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        # Store number of layers for 'all' option
        self.n_layers = self.model.cfg.n_layers

    def get_layer_activations(self, tokens: torch.Tensor, layers: Union[List[int], str] = [19]) -> torch.Tensor:
        """
        Runs a forward pass and returns the residual stream (resid_pre) activations 
        for the specified layers, concatenated along the feature dimension.

        Args:
            tokens (torch.Tensor): A tensor of token IDs, shape [batch, seq_len].
            layers (Union[List[int], str]): List of layer indices to retrieve activations from,
                                            or 'all' to get activations from all layers.

        Returns:
            torch.Tensor: The concatenated activation tensor, shape [batch, seq_len, n_layers * d_model], 
                          stored on the CPU.

        Raises:
            ValueError: If layers is invalid.
            RuntimeError: If any activation could not be retrieved.
        """
        # Handle 'all' option
        if isinstance(layers, str):
            if layers.lower() == 'all':
                layers = list(range(self.n_layers))
            else:
                raise ValueError(
                    f"Invalid layers string: '{layers}'. Use 'all' or a list of integers.")

        if not layers:
            raise ValueError("layers list cannot be empty")

        # Dictionary to store the activations captured by hooks
        activations = {}

        def make_layer_hook(layer_idx: int):
            def layer_hook(resid_pre: torch.Tensor, hook):
                # resid_pre: [batch, seq, d_model] - This is the input to the attention/MLP block.
                # We store a copy of the tensor for external use.
                activations[layer_idx] = resid_pre.detach().clone().cpu()
                return resid_pre
            return layer_hook

        # Create hook points for all requested layers
        fwd_hooks = []
        for layer in layers:
            hook_point = f"blocks.{layer}.hook_resid_pre"
            fwd_hooks.append((hook_point, make_layer_hook(layer)))

        # Run the forward pass with hooks, stopping after the last requested layer
        stop_at_layer = max(layers) + 1

        with torch.no_grad():
            self.model.run_with_hooks(
                tokens.to(self.device),
                fwd_hooks=fwd_hooks,
                stop_at_layer=stop_at_layer
            )

        # Verify all activations were captured
        for layer in layers:
            if layer not in activations:
                raise RuntimeError(
                    f"Could not retrieve activation from hook point blocks.{layer}.hook_resid_pre")

        # Concatenate activations from all layers along the feature dimension
        # Each activation is [batch, seq_len, d_model]
        # Result is [batch, seq_len, n_layers * d_model]
        layer_tensors = [activations[layer] for layer in sorted(layers)]
        concatenated = torch.cat(layer_tensors, dim=-1)

        return concatenated
