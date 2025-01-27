import torch
import numpy as np
from typing import Generator, Optional, Callable
from transformers import PreTrainedModel
import gc

class ActivationBuffer:
    def __init__(
        self,
        text_generator: Generator,
        model: PreTrainedModel,
        submodule: torch.nn.Module,
        n_ctxs: int = 2048,
        ctx_len: int = 128,
        refresh_batch_size: int = 32,
        out_batch_size: int = 125,
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
        
        self.activations = None
        self.current_idx = 0
        
        # Register forward hook
        self.hook = self.submodule.register_forward_hook(self._hook_fn)
        
    def _hook_fn(self, module, input_tensor, output_tensor):
        """Forward hook to capture activations"""
        if self.io == "in":
            self.temp_acts = input_tensor[0].detach()
        else:
            self.temp_acts = output_tensor.detach()
            
    def refresh(self):
        """Refresh activation buffer with new data"""
        self.activations = []
        
        for _ in range(0, self.n_ctxs, self.refresh_batch_size):
            # Get next batch of texts
            texts = []
            for _ in range(self.refresh_batch_size):
                try:
                    text = next(self.text_generator())
                    texts.append(text)
                except StopIteration:
                    break
                    
            if not texts:
                break
                
            # Tokenize and get activations
            inputs = self.model.tokenizer(
                texts,
                max_length=self.ctx_len,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                self.model(**inputs)
                
            self.activations.append(self.temp_acts)
            
        self.activations = torch.cat(self.activations, dim=0)
        self.current_idx = 0
        
        # Clear memory
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.activations is None or self.current_idx + self.out_batch_size > len(self.activations):
            self.refresh()
            
        batch = self.activations[self.current_idx:self.current_idx + self.out_batch_size]
        self.current_idx += self.out_batch_size
        
        return batch.reshape(-1, batch.shape[-1])
