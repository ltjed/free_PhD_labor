from datasets import load_dataset
import torch
import numpy as np

def hf_dataset_to_generator(dataset_name):
    """Convert a HuggingFace dataset to a generator of text samples."""
    dataset = load_dataset(dataset_name, streaming=True)
    
    def generator():
        for item in dataset["train"]:
            if isinstance(item.get("text"), str):
                yield item["text"]
            elif isinstance(item.get("content"), str):
                yield item["content"]
    
    return generator
