from datasets import load_dataset
from typing import Iterator, Dict, Any
import torch

def hf_dataset_to_generator(dataset_name: str) -> Iterator[Dict[str, Any]]:
    """Convert a HuggingFace dataset to a generator of text samples.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        
    Returns:
        Generator yielding text samples
    """
    dataset = load_dataset(dataset_name, streaming=True)
    
    if isinstance(dataset, dict):
        # If dataset has splits, use the train split
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            # Use the first split if train not available
            dataset = list(dataset.values())[0]
            
    for item in dataset:
        if isinstance(item, dict) and 'text' in item:
            yield item['text']
        else:
            # Try to convert the item to string if it's not a dict
            yield str(item)
