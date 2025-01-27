import torch
from typing import Iterator, Optional, Any
from tqdm import tqdm

class ActivationBuffer:
    """Buffer for storing and batching model activations."""
    
    def __init__(
        self,
        text_generator: Iterator[str],
        model: Any,
        submodule: Any,
        n_ctxs: int = 2048,
        ctx_len: int = 128,
        refresh_batch_size: int = 24,
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
            self._refresh_buffer()
            self.buffer_idx = 0
            
        batch = self.buffer[self.buffer_idx:self.buffer_idx + self.out_batch_size]
        self.buffer_idx += self.out_batch_size
        
        return batch
        
    def _refresh_buffer(self):
        """Collect new activations by running model on fresh text."""
        activations = []
        
        for _ in tqdm(range(0, self.n_ctxs, self.refresh_batch_size)):
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
                    
            # Tokenize
            tokens = self.model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.ctx_len,
                return_tensors="pt"
            ).to(self.device)
            
            # Run model and collect activations
            with torch.no_grad():
                def hook(module, input, output):
                    if self.io == "in":
                        acts = input[0]
                    else:
                        acts = output
                    # Ensure acts is detached and moved to device synchronously
                    acts = acts.detach().to(device=self.device)
                    activations.append(acts)
                    
                handle = self.submodule.register_forward_hook(hook)
                try:
                    # Move tokens to device synchronously
                    tokens = {k: v.to(device=self.device) for k, v in tokens.items()}
                    # Run model forward pass
                    self.model(**tokens)
                finally:
                    handle.remove()
                
        # Stack and reshape activations, ensuring device consistency
        self.buffer = torch.cat([
            a.reshape(-1, a.shape[-1]).to(device=self.device)
            for a in activations
        ])
