from datasets import load_dataset
import torch
import numpy as np

def hf_dataset_to_generator(dataset_name, split="train", shuffle=True, seed=42):
    """Convert a HuggingFace dataset to a generator of text samples."""
    try:
        # Load with minimal cache
        dataset = load_dataset(
            dataset_name, 
            split=split,
            trust_remote_code=True,
            cache_dir="/tmp/hf_cache",  # Use temporary directory
            num_proc=1  # Minimize parallel processing
        )
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        
        def text_generator():
            for item in dataset:
                if isinstance(item, dict):
                    # For ag_news dataset
                    if 'text' in item:
                        yield item['text']
                    # For rotten_tomatoes dataset    
                    elif 'text' in item:
                        yield item['text']
                    # Try other common text field names
                    else:
                        for field in ['content', 'document']:
                            if field in item:
                                yield item[field]
                                break
                else:
                    yield str(item)
        
        return text_generator()
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        raise
