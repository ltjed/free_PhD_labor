from datasets import load_dataset
import torch

def hf_dataset_to_generator(dataset_name, split="train"):
    """Convert HuggingFace dataset to a generator of text samples."""
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    def text_generator():
        for item in dataset:
            if isinstance(item, dict) and "text" in item:
                yield item["text"]
            else:
                yield str(item)
                
    return text_generator()
