from datasets import load_dataset
import torch
import numpy as np

def hf_dataset_to_generator(dataset_name, split="train"):
    """Convert a HuggingFace dataset to a generator of text samples."""
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    def generator():
        for item in dataset:
            if isinstance(item, dict) and "text" in item:
                yield item["text"]
            else:
                yield item
                
    return generator()
