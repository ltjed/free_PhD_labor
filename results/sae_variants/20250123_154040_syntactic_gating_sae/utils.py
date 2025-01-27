import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class QuantizedPatternMatcher(nn.Module):
    """Implements quantized pattern matching for activation sequences."""
    def __init__(self, input_dim: int, n_patterns: int, n_quantize_bins: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.n_patterns = n_patterns
        self.n_quantize_bins = n_quantize_bins
        
        # Learnable patterns
        self.patterns = nn.Parameter(torch.randn(n_patterns, input_dim))
        self.quantize_edges = nn.Parameter(torch.linspace(-2, 2, n_quantize_bins-1))
        
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input activations into discrete bins."""
        # x: [batch_size, seq_len, input_dim]
        expanded_edges = self.quantize_edges.view(1, 1, 1, -1)
        expanded_input = x.unsqueeze(-1)  # [batch, seq, dim, 1]
        quantized = (expanded_input > expanded_edges).sum(-1)  # [batch, seq, dim]
        return quantized
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match input sequences against stored patterns."""
        quantized_input = self.quantize(x)  # [batch, seq, dim]
        quantized_patterns = self.quantize(self.patterns.unsqueeze(0))  # [1, n_patterns, dim]
        
        # Compute pattern matches
        matches = torch.zeros(x.shape[0], x.shape[1], self.n_patterns, device=x.device)
        for i in range(self.n_patterns):
            pattern = quantized_patterns[0, i]  # [dim]
            matches[:, :, i] = (quantized_input == pattern).float().mean(dim=-1)
            
        # Get best matching patterns
        best_patterns = matches.max(dim=-1)[1]  # [batch, seq]
        match_scores = matches.max(dim=-1)[0]  # [batch, seq]
        
        return best_patterns, match_scores

class LookupGating(nn.Module):
    """Implements lookup-based gating mechanism."""
    def __init__(self, n_patterns: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_patterns = n_patterns
        self.d_model = d_model
        
        # Pattern-specific gating vectors
        self.gate_vectors = nn.Parameter(torch.randn(n_patterns, d_model))
        self.gate_bias = nn.Parameter(torch.zeros(n_patterns))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, pattern_ids: torch.Tensor, 
                match_scores: torch.Tensor) -> torch.Tensor:
        """Apply pattern-specific gating to input."""
        # x: [batch, seq, d_model]
        # pattern_ids: [batch, seq]
        # match_scores: [batch, seq]
        
        # Get relevant gate vectors
        gates = self.gate_vectors[pattern_ids]  # [batch, seq, d_model]
        bias = self.gate_bias[pattern_ids]  # [batch, seq]
        
        # Apply gating with confidence weighting
        gate_weights = torch.sigmoid(gates * x + bias.unsqueeze(-1))
        gate_weights = gate_weights * match_scores.unsqueeze(-1)
        gate_weights = self.dropout(gate_weights)
        
        return x * gate_weights
