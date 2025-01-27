import torch
import numpy as np
from typing import Iterator, Optional, Union, Callable
from torch import Tensor
import gc

class ActivationBuffer:
    """Buffer for storing and batching activations from a language model."""
    
    def __init__(
        self,
        text_generator: Iterator[str],
        model,
        submodule,
        n_ctxs: int = 2048,
        ctx_len: int = 128,
        refresh_batch_size: int = 32,
        out_batch_size: int = 2048,
        io: str = "out",
        d_submodule: Optional[int] = None,
        device: str = "cuda",
    ):
        self.text_generator = text_generator
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
            try:
                self._refresh_buffer()
                self.buffer_idx = 0
            except StopIteration:
                print("Reached end of dataset, recycling...")
                self.text_generator = iter(self.text_generator)
                self._refresh_buffer()
                self.buffer_idx = 0
                
        if self.buffer is None or len(self.buffer) == 0:
            raise RuntimeError("Failed to get valid activations from buffer")
            
        end_idx = min(self.buffer_idx + self.out_batch_size, len(self.buffer))
        batch = self.buffer[self.buffer_idx:end_idx]
        
        if batch.nelement() == 0:
            raise RuntimeError("Empty batch generated")
            
        self.buffer_idx += self.out_batch_size
        
        return batch
        
    def _refresh_buffer(self):
        """Refresh the activation buffer with new data."""
        activations = []
        total_samples = 0
        
        while total_samples < self.n_ctxs:
            # Get next batch of texts
            texts = []
            for _ in range(self.refresh_batch_size):
                try:
                    text = next(self.text_generator)
                    texts.append(text)
                except StopIteration:
                    if not texts:
                        raise StopIteration
                    break
                    
            # Tokenize texts
            tokens = self.model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.ctx_len,
                return_tensors="pt"
            ).to(self.device)
            
            # Get activations
            with torch.no_grad():
                _ = self.model(**tokens)
                
            # Get activations from hook
            if self.io == "out":
                batch_acts = self.submodule.output
            else:
                batch_acts = self.submodule.input[0]
                
            batch_acts = batch_acts.to(dtype=torch.float32)
            total_samples += batch_acts.shape[0]
            activations.append(batch_acts)
            
            # Clear cache
            del tokens
            torch.cuda.empty_cache()
            gc.collect()
            
        if not activations:
            raise RuntimeError("No activations collected")
            
        # Concatenate all activations
        try:
            self.buffer = torch.cat(activations, dim=0)
            print(f"Buffer refreshed with shape: {self.buffer.shape}")
        except Exception as e:
            print(f"Error concatenating activations: {e}")
            raise
