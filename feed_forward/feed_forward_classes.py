"""
Feed-forward network module for transformer blocks.

Implements a simple two-layer MLP with ReLU activation and dropout,
following the standard transformer feed-forward architecture.
"""

import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed-forward network: a simple linear layer followed by a non-linearity.
    
    Standard transformer feed-forward module that applies:
    1. Linear projection: n_embd -> 4 * n_embd
    2. ReLU activation
    3. Linear projection: 4 * n_embd -> n_embd
    4. Dropout for regularization
    """

    def __init__(self, n_embd, dropout):
        """
        Initialize feed-forward network.
        
        Args:
            n_embd: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            x: Output tensor of shape (B, T, C)
            
        Where:
            B = batch size
            T = sequence length (time dimension)
            C = channels (embedding dimension)
        """
        return self.net(x)  # (B, T, C) -> (B, T, C)