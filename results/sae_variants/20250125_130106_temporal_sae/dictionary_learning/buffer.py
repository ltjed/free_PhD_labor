import torch
from typing import Iterator, Optional, Callable
from transformers import PreTrainedModel
from torch import nn

class ActivationBuffer:
    """Buffer for storing and batching model activations."""
    
    def __init__(
        self,
        text_generator: Iterator[str],
        model: PreTrainedModel,
        submodule: nn.Module,
        n_ctxs: int = 2048,
        ctx_len: int = 128,
        refresh_batch_size: int = 32,
        out_batch_size: int = 2048,
        io: str = "out",
        d_submodule: Optional[int] = None,
        device: str = "cuda",
    ):
        """Initialize activation buffer.
        
        Args:
            text_generator: Iterator yielding text samples
            model: Transformer model
            submodule: Layer to extract activations from
            n_ctxs: Number of contexts to buffer
            ctx_len: Context length for each sample
            refresh_batch_size: Batch size for model forward pass
            out_batch_size: Batch size for returned activations
            io: Whether to capture input ("in") or output ("out") activations
            d_submodule: Dimension of activations if known
            device: Device to store activations on
        """
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
        
        # Initialize buffer
        self.buffer = None
        self.buffer_idx = 0
        
        # Setup activation hooks
        self.hook = None
        self._setup_hooks()
        
    def _setup_hooks(self):
        """Setup hooks to capture activations."""
        activations = []
        
        def hook_fn(module, input, output):
            if self.io == "in":
                activations.append(input[0].detach())
            else:
                activations.append(output.detach())
                
        self.hook = self.submodule.register_forward_hook(hook_fn)
        self._activations = activations
        
    def _refresh_buffer(self):
        """Refresh activation buffer with new samples."""
        self._activations.clear()
        
        # Get new text samples
        texts = []
        for _ in range(self.refresh_batch_size):
            try:
                text = next(self.text_generator)
                texts.append(text)
            except StopIteration:
                break
                
        if not texts:
            raise StopIteration
            
        # Tokenize and get activations
        inputs = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.ctx_len,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            self.model(**inputs)
            
        # Stack activations
        acts = torch.cat(self._activations, dim=0)
        self._activations.clear()
        
        # Reshape to (batch * seq_len, d_model)
        self.buffer = acts.reshape(-1, acts.shape[-1])
        self.buffer_idx = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        """Get next batch of activations."""
        if self.buffer is None or self.buffer_idx >= len(self.buffer):
            self._refresh_buffer()
            
        start_idx = self.buffer_idx
        end_idx = min(start_idx + self.out_batch_size, len(self.buffer))
        batch = self.buffer[start_idx:end_idx]
        
        self.buffer_idx = end_idx
        
        if len(batch) < self.out_batch_size:
            self._refresh_buffer()
            remaining = self.out_batch_size - len(batch)
            batch = torch.cat([batch, self.buffer[:remaining]])
            self.buffer_idx = remaining
            
        return batch
