"""General utility functions for the SAE project."""

import os
import json
import torch
import numpy as np
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save experiment results to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy format
    np.save(os.path.join(output_dir, "results.npy"), results)
    
    # Also save as JSON for easier inspection
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_results(output_dir: str) -> Dict[str, Any]:
    """Load experiment results from the specified directory."""
    results_path = os.path.join(output_dir, "results.npy")
    if os.path.exists(results_path):
        return np.load(results_path, allow_pickle=True).item()
    else:
        raise FileNotFoundError(f"No results found in {output_dir}")

def setup_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create and return path to experiment directory with incrementing run number."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{timestamp}_{experiment_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir
