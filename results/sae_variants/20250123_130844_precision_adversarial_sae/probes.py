import torch
import torch.nn as nn

class SimpleProbe(nn.Module):
    """Simple linear probe for classification"""
    def __init__(self, input_dim=2304):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

def load_probes():
    """Load pre-trained gender and profession probes"""
    gender_probe = SimpleProbe()
    prof_probe = SimpleProbe()
    # Initialize with random weights for now
    # In practice, these would be loaded from pre-trained checkpoints
    return gender_probe, prof_probe
