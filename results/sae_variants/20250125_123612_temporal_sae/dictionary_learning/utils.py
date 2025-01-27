from datasets import load_dataset
from typing import Generator, Any

def hf_dataset_to_generator(dataset_name: str) -> Generator[str, None, None]:
    """Convert a HuggingFace dataset to a generator of text strings."""
    dataset = load_dataset(dataset_name, streaming=True)
    
    for item in dataset["train"]:
        if isinstance(item.get("text"), str):
            yield item["text"]
