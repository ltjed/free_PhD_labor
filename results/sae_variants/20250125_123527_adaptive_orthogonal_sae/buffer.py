import torch
from typing import Iterator, Callable, Optional
from tqdm import tqdm

class ActivationBuffer:
    """Buffer for storing and batching model activations."""
    
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
        
    def _get_hook(self):
        activations = []
        
        def hook(module, input, output):
            if self.io == "in":
                acts = input[0]
            else:
                if isinstance(output, tuple):
                    acts = output[0]  # Take first element if tuple
                else:
                    acts = output
            activations.append(acts.detach().to(self.device))
            
        return hook, activations
        
    def refresh(self):
        """Refresh buffer with new activations."""
        hook, activations = self._get_hook()
        handle = self.submodule.register_forward_hook(hook)
        
        try:
            for _ in range(self.refresh_batch_size):
                text = next(self.text_generator)
                if not hasattr(self.model, 'tokenizer'):
                    from transformers import AutoTokenizer
                    self.model.tokenizer = AutoTokenizer.from_pretrained(
                        self.model.config.name_or_path, 
                        trust_remote_code=True
                    )
                
                if self.model.tokenizer.pad_token is None:
                    self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
                inputs = self.model.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.ctx_len,
                    truncation=True,
                    padding="max_length",
                    stride=self.ctx_len - 1  # Adjust stride to be less than effective max length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    self.model(**inputs)
                    
        except StopIteration:
            print("Reached end of dataset")
            
        handle.remove()
        
        # Stack and reshape activations
        acts = torch.cat(activations, dim=0)
        acts = acts.reshape(-1, acts.shape[-1])
        
        if self.buffer is None:
            self.buffer = acts
        else:
            self.buffer = torch.cat([self.buffer, acts], dim=0)
            
        self.buffer_idx = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.buffer is None or self.buffer_idx + self.out_batch_size > len(self.buffer):
            self.refresh()
            
        batch = self.buffer[self.buffer_idx:self.buffer_idx + self.out_batch_size]
        self.buffer_idx += self.out_batch_size
        
        return batch
