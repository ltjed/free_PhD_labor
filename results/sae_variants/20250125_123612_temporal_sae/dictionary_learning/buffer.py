import torch
from typing import Generator, Iterator, Optional
from nnsight import LanguageModel
from transformers import PreTrainedModel

class ActivationBuffer:
    """Buffer for storing and batching model activations."""
    
    def __init__(
        self,
        text_generator: Generator[str, None, None],
        model: LanguageModel,
        submodule: PreTrainedModel,
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
            text_generator: Generator yielding text strings
            model: Language model to get activations from
            submodule: Model submodule to hook
            n_ctxs: Number of contexts to buffer
            ctx_len: Context length for each text sample
            refresh_batch_size: Batch size for model forward pass
            out_batch_size: Batch size for returned activations
            io: Whether to capture "in" or "out" activations
            d_submodule: Dimension of submodule activations
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
        self.d_submodule = d_submodule or model.config.hidden_size
        self.device = device
        
        # Initialize empty buffer
        self.buffer = torch.zeros(
            (n_ctxs * ctx_len, self.d_submodule),
            device=device,
            dtype=torch.bfloat16  # Use bfloat16 to reduce memory usage
        )
        self.buffer_idx = 0
        self.n_stored = 0
        
    def refresh(self) -> None:
        """Refresh buffer with new activations from model."""
        n_batches = self.n_ctxs // self.refresh_batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                # Get next batch of texts
                texts = []
                for _ in range(self.refresh_batch_size):
                    try:
                        texts.append(next(self.text_generator))
                    except StopIteration:
                        # Restart generator if exhausted
                        self.text_generator = iter(self.text_generator)
                        texts.append(next(self.text_generator))
                
                # Get activations from model
                with torch.no_grad():
                    with self.model.generate(max_new_tokens=1) as executor:
                        with executor.invoke(texts):
                            # Set up activation capture
                            if self.io == "in":
                                hook_point = self.submodule.input.save()
                            else:
                                hook_point = self.submodule.output.save()
                    
                    # Access saved activations after invoke context
                    acts = hook_point.value
                    # Handle tuple case - take first element if tuple
                    if isinstance(acts, tuple):
                        acts = acts[0]
                    # Ensure we have the right shape
                    acts = acts.reshape(-1, self.d_submodule)
                
                # Store in buffer
                start_idx = i * self.refresh_batch_size * self.ctx_len
                end_idx = (i + 1) * self.refresh_batch_size * self.ctx_len
                # Ensure proper reshaping by explicitly using batch size and context length
                reshaped_acts = acts.reshape(self.refresh_batch_size * self.ctx_len, self.d_submodule)
                self.buffer[start_idx:end_idx] = reshaped_acts
        
        self.n_stored = self.n_ctxs * self.ctx_len
        self.buffer_idx = 0
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        return self
        
    def __next__(self) -> torch.Tensor:
        """Get next batch of activations."""
        if self.buffer_idx + self.out_batch_size > self.n_stored:
            self.refresh()
            
        batch = self.buffer[self.buffer_idx:self.buffer_idx + self.out_batch_size]
        self.buffer_idx += self.out_batch_size
        return batch
