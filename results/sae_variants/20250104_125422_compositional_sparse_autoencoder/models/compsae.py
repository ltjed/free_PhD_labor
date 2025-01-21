import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass 
class CompSAEConfig:
    d_in: int  # Input dimension
    d_f: int = 256  # Feature extractor dimension
    d_c: int = 128  # Composition network dimension
    n_heads: int = 8  # Number of attention heads
    sparsity_k: int = 32  # Sparsity bottleneck
    dropout: float = 0.1

class CompSAE(nn.Module):
    """Compositional Sparse Autoencoder with two-stream architecture."""
    def __init__(self, config: CompSAEConfig):
        super().__init__()
        self.config = config

        # Feature extractor network
        self.W_enc = nn.Parameter(torch.zeros(config.d_in, config.d_f))
        self.b_enc = nn.Parameter(torch.zeros(config.d_f))
        
        # Composition network
        self.mha = nn.MultiheadAttention(
            embed_dim=config.d_c,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Projection layers
        self.proj_f = nn.Linear(config.d_f, config.d_c)
        self.proj_c = nn.Linear(config.d_c, config.d_f)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(config.d_f * 2, config.d_f),
            nn.Sigmoid()
        )
        
        # Decoder
        self.W_dec = nn.Parameter(torch.zeros(config.d_f, config.d_in))
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))
        
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()

    def init_weights(self):
        # Initialize weights with small random values
        nn.init.kaiming_normal_(self.W_enc)
        nn.init.kaiming_normal_(self.W_dec)
        
    def encode(self, x):
        # Feature extraction
        f = torch.relu(x @ self.W_enc + self.b_enc)
        
        # Project features to composition space
        c = self.proj_f(f)
        
        # Multi-head attention for feature composition
        c_attn, _ = self.mha(c, c, c)
        c_attn = self.dropout(c_attn)
        
        # Project back to feature space
        f_composed = self.proj_c(c_attn)
        
        # Gating mechanism for residual connection
        gate = self.gate(torch.cat([f, f_composed], dim=-1))
        f_final = gate * f + (1 - gate) * f_composed
        
        # Sparsity bottleneck
        topk_values, _ = torch.topk(f_final, self.config.sparsity_k, dim=-1)
        threshold = topk_values[..., -1:]
        return torch.relu(f_final - threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        return x_hat
