"""
Self-attention mechanism implementation for transformers.

Contains Head (single attention head) and MultiHeadAttention classes
that implement scaled dot-product attention with causal masking for
autoregressive language modeling.
"""

# Third-party imports
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Single head of self-attention. This is a single attention head that is used to compute the attention over the input sequence.

    Implements scaled dot-product attention with causal masking. Causal masking is used to prevent the model from attending to future tokens in the sequence.
    Each head independently computes attention over the input sequence. This is done by projecting the input sequence into a key, query, and value space.
    The key, query, and value are then used to compute the attention scores. The attention scores are then used to compute the output of the attention head.
    The output of the attention head is then used to compute the output of the multi-head attention layer.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        """
        Initialize attention head.

        Args:
            n_embd: Embedding dimension - number of features in the input data. Example: 256 for a 256-dimensional embedding.
            head_size: Dimension of each attention head - number of features in the output data. Example: 64 for a 64-dimensional head.
            block_size: Maximum sequence length (for causal mask) - number of tokens in the sequence. Example: 1024 for a 1024-token sequence.
            dropout: Dropout probability - This is used to prevent overfitting by randomly dropping out some of the input features.
            Example: 0.1 for a 10% dropout rate.
        """
        super().__init__()
        self.key: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.query: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.value: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention head. This is the main function that is used to compute the attention over the input sequence. The attention head is used to compute the attention over the input sequence.

        Args:
            x: Input tensor of shape (B, T, C) - batch size, sequence length, and embedding dimension.
            Example: (1, 1024, 256) for a batch size of 1, a sequence length of 1024, and a 256-dimensional embedding.

        Returns:
            out: Output tensor of shape (B, T, head_size) - batch size, sequence length, and head size. Head size is the dimension/feature size of the output feature vector.
            Example: (1, 1024, 64) for a batch size of 1, a sequence length of 1024, and a 64-dimensional head (64 features/dimensions).
        """
        B: int = x.shape[0]  # Extract batch size # noqa: F841
        T: int = x.shape[1]  # Extract sequence length
        C: int = x.shape[2]  # Extract embedding dimension

        # Compute key, query, value projections - This is done by projecting the input sequence into a key, query, and value space.
        k: torch.Tensor = self.key(x)  # (B, T, head_size)
        q: torch.Tensor = self.query(x)  # (B, T, head_size)
        v: torch.Tensor = self.value(x)  # (B, T, head_size)

        # Compute attention scores ("affinities") - This is done by computing the attention scores between the key and query.
        # Scaled dot-product: Q @ K^T / sqrt(head_size)
        weight: torch.Tensor = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T) # The extra math is needed to scale the attention scores. This means that the attention scores are not too large or too small.

        # Apply causal mask (lower triangular) to prevent looking at future tokens - This is done to prevent the model from attending to future tokens in the sequence.
        weight: torch.Tensor = weight.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)

        # Convert scores to probabilities via softmax - This is done to convert the attention scores to probabilities.
        weight: torch.Tensor = F.softmax(weight, dim=-1)  # (B, T, T)
        weight: torch.Tensor = self.dropout(weight)

        # Perform the weighted aggregation of the values - This is done to aggregate the values based on the attention scores.
        out: torch.Tensor = (
            weight @ v
        )  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.

    Runs multiple attention heads in parallel and concatenates their outputs,
    then applies a linear projection to combine them.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        block_size: int,
        dropout: float,
    ):
        """
        Initialize multi-head attention.

        Args:
            num_heads: Number of parallel attention heads - number of attention heads to run in parallel.
            head_size: Dimension of each attention head - number of features in the output data.
            n_embd: Embedding dimension - number of features in the input data.
            block_size: Maximum sequence length (for causal mask) - number of tokens in the sequence.
            dropout: Dropout probability - This is used to prevent overfitting by randomly dropping out some of the input features.
            Example: 0.1 for a 10% dropout rate.
        """
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj: nn.Linear = nn.Linear(num_heads * head_size, n_embd)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (B, T, C) - batch size, sequence length, and embedding dimension.
            Example: (1, 1024, 256) for a batch size of 1, a sequence length of 1024, and a 256-dimensional embedding.

        Returns:
            out: Output tensor of shape (B, T, C) - batch size, sequence length, and embedding dimension.
            Example: (1, 1024, 256) for a batch size of 1, a sequence length of 1024, and a 256-dimensional embedding.

        """
        # Concatenate outputs from all heads: (B, T, head_size) * num_heads -> (B, T, C) - This is done to concatenate the outputs from all the attention heads.
        out: torch.Tensor = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply output projection and dropout - This is done to project the concatenated outputs into the embedding dimension.
        out: torch.Tensor = self.dropout(self.proj(out))  # (B, T, C) -> (B, T, C)
        return out
