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
    Single head of self-attention.
    
    Implements scaled dot-product attention with causal masking.
    Each head independently computes attention over the input sequence.
    """

    def __init__(self, n_embd, head_size, block_size, dropout):
        """
        Initialize attention head.
        
        Args:
            n_embd: Embedding dimension
            head_size: Dimension of each attention head
            block_size: Maximum sequence length (for causal mask)
            dropout: Dropout probability
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through attention head.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            out: Output tensor of shape (B, T, head_size)
            
        Where:
            B = batch size
            T = sequence length (time dimension)
            C = channels (embedding dimension)
        """
        B, T, C = x.shape  # Extract batch, time, channels
        
        # Compute key, query, value projections
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # Scaled dot-product: Q @ K^T / sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        # Apply causal mask (lower triangular) to prevent looking at future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        
        # Convert scores to probabilities via softmax
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    
    Runs multiple attention heads in parallel and concatenates their outputs,
    then applies a linear projection to combine them.
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        """
        Initialize multi-head attention.
        
        Args:
            num_heads: Number of parallel attention heads
            head_size: Dimension of each attention head
            n_embd: Embedding dimension (should equal num_heads * head_size)
            block_size: Maximum sequence length (for causal mask)
            dropout: Dropout probability
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            out: Output tensor of shape (B, T, C)
            
        Where:
            B = batch size
            T = sequence length (time dimension)
            C = channels (embedding dimension)
        """
        # Concatenate outputs from all heads: (B, T, head_size) * num_heads -> (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply output projection and dropout
        out = self.dropout(self.proj(out))  # (B, T, C) -> (B, T, C)
        return out