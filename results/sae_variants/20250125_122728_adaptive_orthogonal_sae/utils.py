import torch
from datasets import load_dataset
from typing import Iterator, Any
import numpy as np

def hf_dataset_to_generator(dataset_name: str) -> Iterator[str]:
    """Convert HuggingFace dataset to text generator."""
    dataset = load_dataset(dataset_name, streaming=True)
    for item in dataset["train"]:
        if isinstance(item["text"], str):
            yield item["text"]

class ActivationBuffer:
    """Buffer for storing and batching model activations."""
    def __init__(
        self,
        generator: Iterator[str],
        model: Any,
        submodule: Any,
        n_ctxs: int = 2048,
        ctx_len: int = 128,
        refresh_batch_size: int = 24,
        out_batch_size: int = 2048,
        io: str = "out",
        d_submodule: int = None,
        device: str = "cuda",
    ):
        self.generator = generator
        self.model = model
        self.submodule = submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.io = io
        self.d_submodule = d_submodule
        self.device = device
        
        self.buffer = None
        self.buffer_idx = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.buffer is None or self.buffer_idx >= len(self.buffer):
            self._refresh_buffer()
            self.buffer_idx = 0
            
        batch = self.buffer[self.buffer_idx:self.buffer_idx + self.out_batch_size]
        self.buffer_idx += self.out_batch_size
        
        return batch.to(self.device)
        
    def _refresh_buffer(self):
        """Get new batch of activations from model."""
        texts = []
        for _ in range(self.refresh_batch_size):
            try:
                text = next(self.generator)
                texts.append(text[:self.ctx_len])
            except StopIteration:
                break
                
        if not texts:
            raise StopIteration
            
        with torch.no_grad():
            inputs = self.model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get activations
            acts = []
            def hook_fn(module, input, output):
                if self.io == "in":
                    acts.append(input[0].detach())
                else:
                    acts.append(output.detach())
                    
            handle = self.submodule.register_forward_hook(hook_fn)
            self.model(**inputs)
            handle.remove()
            
            # Process activations
            acts = torch.cat(acts, dim=0)
            self.buffer = acts.reshape(-1, acts.shape[-1])
