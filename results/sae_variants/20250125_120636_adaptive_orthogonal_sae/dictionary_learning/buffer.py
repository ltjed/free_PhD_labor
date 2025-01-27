import torch
from typing import Generator, Any, Optional
from nnsight import LanguageModel
import numpy as np

class ActivationBuffer:
    """Buffer for storing and batching model activations."""
    

    def __init__(
        self,
        generator: Generator[str, None, None],
        model: LanguageModel,
        submodule: Any,
        n_ctxs: int = 2048,
        ctx_len: int = 128,
        refresh_batch_size: int = 32,
        out_batch_size: int = 2048,
        io: str = "out",
        d_submodule: Optional[int] = None,
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
        
        self.buffer_idx = 0
        self.buffer = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.buffer is None or self.buffer_idx >= len(self.buffer):
            self._refresh_buffer()
            self.buffer_idx = 0
            
        start_idx = self.buffer_idx
        end_idx = min(start_idx + self.out_batch_size, len(self.buffer))
        self.buffer_idx = end_idx
        
        return self.buffer[start_idx:end_idx]
    
    def _refresh_buffer(self):
        """Refresh the activation buffer with new data."""
        texts = []
        for _ in range(self.refresh_batch_size):
            try:
                text = next(self.generator)
                texts.append(text)
            except StopIteration:
                if not texts:
                    raise StopIteration
                break
                
        with torch.no_grad():
            tokens = self.model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.ctx_len,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            if self.io == "out":
                activations = outputs.hidden_states[self.submodule]
            else:
                activations = outputs.hidden_states[self.submodule - 1]
                
            # Reshape activations to (batch * seq_len, d_model)
            self.buffer = activations.reshape(-1, activations.size(-1))
