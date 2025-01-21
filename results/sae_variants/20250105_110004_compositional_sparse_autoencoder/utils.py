import torch
import random
import numpy as np

def setup_environment(seed=42):
    """Set up the computing environment and random seeds."""
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return device

def str_to_dtype(dtype_str):
    """Convert string dtype specification to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)
