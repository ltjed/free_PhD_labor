from datasets import load_dataset
import torch
import numpy as np

def hf_dataset_to_generator(dataset_name, split="train"):
    """Convert a HuggingFace dataset to a generator of text samples.
    
    Args:
        dataset_name (str): Name of the dataset on HuggingFace
        split (str): Dataset split to use (default: "train")
        
    Returns:
        generator: Generator yielding text samples
    """
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    def text_generator():
        for item in dataset:
            if isinstance(item, dict):
                # Try common text field names
                for field in ['text', 'content', 'document']:
                    if field in item:
                        yield item[field]
                        break
            else:
                yield str(item)
                
    return text_generator()
