"""
Bigram Language Model with LoRA support.

Extends BigramLanguageModel to support LoRA (Low-Rank Adaptation)
for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer_core.block import Block
from lora.lora_module import apply_lora_to_linear, count_lora_parameters


class BigramLanguageModelLoRA(nn.Module):
    """
    Bigram Language Model with LoRA support for efficient fine-tuning.

    This model can be used in two modes:
    1. Full fine-tuning: Set use_lora=False (trains all parameters)
    2. LoRA fine-tuning: Set use_lora=True (only trains LoRA adapters, ~0.1-1% of params)
    """

    def __init__(
        self,
        vocab_size,
        n_embd,
        block_size,
        device,
        dropout,
        n_head,
        n_layer,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        """
        Initialize Bigram Language Model with optional LoRA support.

        Args:
            vocab_size: Size of the vocabulary
            n_embd: Embedding dimension
            block_size: Maximum sequence length
            device: Device to run on ('cpu', 'cuda', 'mps')
            dropout: Dropout probability
            n_head: Number of attention heads
            n_layer: Number of transformer layers
            use_lora: If True, use LoRA adapters (freeze base model)
            lora_rank: Rank of LoRA matrices (typically 4-32)
            lora_alpha: LoRA scaling factor (typically rank * 2)
            lora_dropout: Dropout probability for LoRA path
        """
        super().__init__()
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.block_size = block_size
        self.device = device

        # Embedding layers (always trainable, even with LoRA)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.blocks.append(nn.LayerNorm(n_embd))

        # Language model head
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora()

    def _apply_lora(self):
        """
        Apply LoRA adapters to all linear layers in the model.

        LoRA is applied to:
        - Attention layers: key, query, value, projection
        - Feed-forward layers: both linear layers
        - Language model head
        """
        # Apply LoRA to attention layers in each block
        for block in self.blocks:
            if hasattr(block, "sa"):
                # Multi-head attention
                for head in block.sa.heads:
                    # Apply LoRA to key, query, value projections
                    head.key = apply_lora_to_linear(
                        head.key,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )
                    head.query = apply_lora_to_linear(
                        head.query,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )
                    head.value = apply_lora_to_linear(
                        head.value,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )

                # Apply LoRA to attention output projection
                block.sa.proj = apply_lora_to_linear(
                    block.sa.proj,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                )

            if hasattr(block, "ffwd"):
                # Feed-forward network
                # Apply LoRA to both linear layers in the feed-forward network
                # The feed-forward is: Linear(n_embd -> 4*n_embd) -> ReLU -> Linear(4*n_embd -> n_embd)
                if isinstance(block.ffwd.net[0], nn.Linear):
                    block.ffwd.net[0] = apply_lora_to_linear(
                        block.ffwd.net[0],
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )
                if isinstance(block.ffwd.net[2], nn.Linear):
                    block.ffwd.net[2] = apply_lora_to_linear(
                        block.ffwd.net[2],
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )

        # Apply LoRA to language model head
        self.lm_head = apply_lora_to_linear(
            self.lm_head,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )

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

        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        position_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = token_emb + position_emb
        x = self.blocks(x)  # (B, T, C). Apply multiple blocks of the transformer
        logits = self.lm_head(x)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy loss function
            B, T, C = logits.shape  # (B, T, C)
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
            idx_cond = idx[:, -self.block_size :]

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

    def get_parameter_info(self):
        """
        Get information about model parameters (useful for LoRA).

        Returns:
            Dictionary with parameter counts and percentages, including
            breakdown of LoRA vs embedding parameters
        """
        info = count_lora_parameters(self)

        # Count embedding parameters separately (they're trainable but not LoRA)
        embedding_params = 0
        if hasattr(self, "token_embedding_table"):
            embedding_params += sum(
                p.numel() for p in self.token_embedding_table.parameters()
            )
        if hasattr(self, "position_embedding_table"):
            embedding_params += sum(
                p.numel() for p in self.position_embedding_table.parameters()
            )

        # Calculate LoRA-only parameters (excluding embeddings)
        lora_only_params = info["trainable"] - embedding_params

        # Add breakdown to info
        info["embedding_params"] = embedding_params
        info["lora_only_params"] = lora_only_params
        info["lora_only_percentage"] = (
            (lora_only_params / info["total"] * 100) if info["total"] > 0 else 0.0
        )
        info["embedding_percentage"] = (
            (embedding_params / info["total"] * 100) if info["total"] > 0 else 0.0
        )

        return info
