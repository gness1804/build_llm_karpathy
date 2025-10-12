"""
Bigram Language Model

A simple bigram language model that predicts the next character
based only on the current character using a lookup table.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer_core.block import Block


class BigramLanguageModel(nn.Module):
    """
    Simple bigram language model that predicts the next character
    based only on the current character using a lookup table.
    """
    
    def __init__(self, vocab_size, n_embd, block_size, device, dropout, n_head, n_layer):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.blocks.append(nn.LayerNorm(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.device = device
    
    def forward(self, idx, targets=None):
        """
        Compute logits (raw predictions) for what could come next after each position
        
        Args:
            idx: Input tensor of shape (B, T) containing token indices
            targets: Optional target tensor of shape (B, T)
            
        Returns:
            logits: Predictions of shape (B, T, C) or (B*T, C) if targets provided
            loss: Cross-entropy loss if targets provided, None otherwise
            
        Where:
            B = batch size
            T = sequence length (tokens in a chunk) / time dimension
            C = channels (number of features/dimensions per token, equals vocab_size here)
        """

        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) # (B, T, C)
        position_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = token_emb + position_emb
        x = self.blocks(x) # (B, T, C). Apply multiple blocks of the transformer
        logits = self.lm_head(x) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy loss function
            B, T, C = logits.shape # (B, T, C)
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens by repeatedly predicting and sampling
        
        Args:
            idx: Starting context of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            
        Returns:
            idx: Extended sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to last block_size tokens to avoid index errors
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions for the cropped context
            logits, loss = self(idx_cond)
            
            # Focus only on the last time step: (B, T, C) -> (B, C)
            logits = logits[:, -1, :]
            
            # Apply softmax to convert to probabilities: (B, C)
            probs = F.softmax(logits, dim=-1)
            
            # Sample one token from the distribution: (B, C) -> (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to the running sequence: (B, T) -> (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

