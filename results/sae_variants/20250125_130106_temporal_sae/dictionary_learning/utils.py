from datasets import load_dataset
import torch

def hf_dataset_to_generator(dataset_name, split="train", max_samples=1000):
    """Convert a HuggingFace dataset to a generator of text samples.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split to use (default: "train")
        max_samples: Maximum number of samples to load (default: 1000)
        
    Returns:
        Generator yielding text samples
    """
    try:
        # Use smaller dataset to avoid disk quota issues
        dataset_name = "tiny_shakespeare"
        dataset = load_dataset(dataset_name, split=split)
        
        def generator():
            count = 0
            for item in dataset:
                if count >= max_samples:
                    break
                    
                if isinstance(item, dict):
                    # Try common text field names
                    for field in ['text', 'content', 'document']:
                        if field in item:
                            yield item[field]
                            count += 1
                            break
                else:
                    yield str(item)
                    count += 1
                    
        return generator()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to simple text if dataset loading fails
        def fallback_generator():
            text = "The quick brown fox jumps over the lazy dog. " * 100
            for i in range(max_samples):
                yield text
        return fallback_generator()
