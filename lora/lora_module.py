"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning.

LoRA adds trainable low-rank matrices to existing linear layers,
allowing efficient fine-tuning by only training a small fraction
of the model parameters.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.

    Wraps a base linear layer and adds low-rank adaptation:
    output = base_layer(x) + (x @ A^T) @ B^T * scaling

    Where:
    - base_layer: Original linear layer (frozen)
    - A: Low-rank matrix (rank, in_features) - trainable
    - B: Low-rank matrix (out_features, rank) - trainable
    - scaling: LoRA alpha / rank (scaling factor)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Initialize LoRA adapter for a linear layer.

        Args:
            base_layer: The original linear layer to adapt (will be frozen)
            rank: Rank of the low-rank matrices (typically 4-32)
            alpha: LoRA scaling factor (typically rank * 2)
            dropout: Dropout probability for LoRA path
        """
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Get dimensions from base layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Initialize LoRA matrices
        # A: (rank, in_features) - initialized with Kaiming uniform
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: (out_features, rank) - initialized to zero
        # This ensures LoRA starts with zero contribution (output = base_layer)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base layer output (frozen) - use no_grad context to skip gradient tracking
        # Note: We still need the output, but we can skip gradient computation overhead
        with torch.no_grad():
            base_output = self.base_layer(x)

        # LoRA adaptation: x @ A^T @ B^T * scaling
        # Optimized: combine the two matrix multiplications
        # x: (..., in_features)
        # A: (rank, in_features) -> A^T: (in_features, rank)
        # B: (out_features, rank) -> B^T: (rank, out_features)
        # Result: (..., out_features)
        x_drop = self.dropout(x)
        # More efficient: (x @ A^T) @ B^T instead of separate operations
        lora_output = (x_drop @ self.lora_A.T) @ self.lora_B.T * self.scaling

        return base_output + lora_output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


def apply_lora_to_linear(
    layer: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRALinear:
    """
    Convenience function to wrap a linear layer with LoRA.

    Args:
        layer: Linear layer to wrap
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability

    Returns:
        LoRALinear wrapper around the original layer
    """
    return LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)


def count_lora_parameters(model: nn.Module) -> dict:
    """
    Count trainable parameters in a model with LoRA.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts:
        - total: Total parameters
        - trainable: Trainable parameters (LoRA only)
        - frozen: Frozen parameters (base model)
        - lora_percentage: Percentage of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "lora_percentage": (
            (trainable_params / total_params * 100) if total_params > 0 else 0.0
        ),
    }
