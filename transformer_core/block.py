"""
Transformer block implementation.

Contains the Block class which combines self-attention and feed-forward
layers with residual connections and layer normalization following the
pre-norm transformer architecture.
"""

# Third-party imports
import torch.nn as nn

# Local imports
from self_attention.self_attention_classes import MultiHeadAttention
from feed_forward.feed_forward_classes import FeedForward


class Block(nn.Module):
    """
    Transformer block: communication followed by computation.

    A complete transformer block consisting of:
    1. Multi-head self-attention with residual connection (pre-norm)
    2. Feed-forward network with residual connection (pre-norm)

    Uses pre-norm architecture: LayerNorm -> Attention/FFN -> Residual
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initialize transformer block.

        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            block_size: Maximum sequence length (for causal masking)
            dropout: Dropout probability
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            x: Output tensor of shape (B, T, C)

        Where:
            B = batch size
            T = sequence length (time dimension)
            C = channels (embedding dimension)
        """
        # Pre-norm architecture: LayerNorm -> Attention -> Residual
        x = x + self.sa(self.ln1(x))  # (B, T, C) -> (B, T, C)
        # Pre-norm architecture: LayerNorm -> FeedForward -> Residual
        x = x + self.ffwd(self.ln2(x))  # (B, T, C) -> (B, T, C)
        return x
