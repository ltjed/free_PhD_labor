import torch
import torch.nn as nn
from typing import Optional
import custom_saes.custom_sae_config as sae_config


class IdentitySAE(nn.Module):
    def __init__(
        self,
        model_name: str,
        d_model: int,
        hook_layer: int,
        hook_name: Optional[str] = None,
        context_size: int = 128,
    ):
        super().__init__()

        # Initialize W_enc and W_dec as identity matrices
        self.W_enc = nn.Parameter(torch.eye(d_model))
        self.W_dec = nn.Parameter(torch.eye(d_model))
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        # required only for the core/main.py SAE evaluation
        self.b_enc = nn.Parameter(torch.zeros(d_model))

        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        # Initialize the configuration dataclass
        self.cfg = sae_config.CustomSAEConfig(
            model_name,
            d_in=d_model,
            d_sae=d_model,
            hook_name=hook_name,
            hook_layer=hook_layer,
            context_size=context_size,
        )

    def encode(self, input_acts: torch.Tensor):
        acts = input_acts @ self.W_enc
        return acts

    def decode(self, acts: torch.Tensor):
        return acts @ self.W_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    # required as we have device and dtype class attributes
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Update the device and dtype attributes based on the first parameter
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        # Update device and dtype if they were provided
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self


if __name__ == "__main__":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_name = "pythia-70m-deduped"
    hook_layer = 3
    d_model = 512

    identity = IdentitySAE(model_name, d_model, hook_layer).to(device=device)
    test_input = torch.randn(1, 128, d_model, device=device, dtype=torch.float32)

    encoded = identity.encode(test_input)

    test_output = identity.decode(encoded)

    print(f"L0: {(encoded != 0).sum() / 128}")

    print(f"Diff: {torch.abs(test_input - test_output).mean()}")

    assert torch.equal(test_input, test_output)
