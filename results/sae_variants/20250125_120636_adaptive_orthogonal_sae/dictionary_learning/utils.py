from datasets import load_dataset
from typing import Generator, Any
from nnsight import LanguageModel
import torch

def hf_dataset_to_generator(dataset_name: str) -> Generator[str, None, None]:
    """Convert a HuggingFace dataset to a generator of text strings."""
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    for item in dataset:
        if isinstance(item, dict) and "text" in item:
            yield item["text"]
        else:
            yield str(item)
