"""
GPT-2 Model Wrapper for Fine-Tuning

Provides a wrapper around HuggingFace's GPT-2 model that:
1. Loads pre-trained GPT-2 models
2. Supports fine-tuning with or without LoRA
3. Maintains compatibility with the training script interface
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional

# Import LoRA support if available
try:
    from lora.lora_module import apply_lora_to_linear, count_lora_parameters

    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print(
        "⚠️  LoRA module not available. Install LoRA support for efficient fine-tuning."
    )


class GPT2Wrapper(nn.Module):
    """
    Wrapper around HuggingFace GPT-2 model for fine-tuning.

    Provides a consistent interface with BigramLanguageModel for training.
    Supports optional LoRA fine-tuning for efficiency.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        device: str = "cpu",
    ):
        """
        Initialize GPT-2 model wrapper.

        Args:
            model_name: HuggingFace model name ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
            use_lora: If True, apply LoRA adapters (requires LoRA module)
            lora_rank: LoRA rank (if use_lora=True)
            lora_alpha: LoRA alpha scaling factor (if use_lora=True)
            lora_dropout: LoRA dropout (if use_lora=True)
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        self.device = device

        # Load pre-trained GPT-2 model and tokenizer
        print(f"📥 Loading {model_name} from HuggingFace...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Set pad_token to eos_token (GPT-2 doesn't have a pad token by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move model to device
        self.model.to(device)

        # Apply LoRA if requested
        if use_lora:
            if not LORA_AVAILABLE:
                raise ImportError(
                    "LoRA module not available. Install LoRA support or set use_lora=False"
                )
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)
            print(f"✅ LoRA adapters applied (rank={lora_rank}, alpha={lora_alpha})")
        else:
            print(
                f"✅ Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters"
            )

    def _apply_lora(self, rank: int, alpha: float, dropout: float):
        """
        Apply LoRA adapters to GPT-2 model.

        GPT-2 architecture uses:
        - Attention: c_attn (combined q, k, v) and c_proj (output projection)
        - Feed-forward: c_fc (first linear) and c_proj (second linear)
        - Output: lm_head (language model head)
        """
        # First, freeze ALL parameters in the base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Collect modules to wrap (don't modify during iteration)
        # GPT-2 uses Conv1D instead of nn.Linear, so we check for modules with weight attribute
        modules_to_wrap = []

        # Import Conv1D to check for it
        try:
            from transformers.pytorch_utils import Conv1D

            CONV1D_AVAILABLE = True
        except ImportError:
            CONV1D_AVAILABLE = False
            Conv1D = None

        for name, module in self.model.named_modules():
            # Check if it's a linear-like layer (Linear or Conv1D)
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = CONV1D_AVAILABLE and isinstance(module, Conv1D)

            if is_linear or is_conv1d:
                # GPT-2 uses c_attn which is a single Conv1D layer combining q, k, v
                if "attn.c_attn" in name:
                    modules_to_wrap.append((name, module))

                # Attention output projection
                elif "attn.c_proj" in name:
                    modules_to_wrap.append((name, module))

                # Feed-forward first layer
                elif "mlp.c_fc" in name:
                    modules_to_wrap.append((name, module))

                # Feed-forward second layer
                elif "mlp.c_proj" in name:
                    modules_to_wrap.append((name, module))

        # Apply LoRA to collected modules
        for name, module in modules_to_wrap:
            # Navigate to parent and replace module
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            setattr(
                parent,
                attr_name,
                apply_lora_to_linear(module, rank=rank, alpha=alpha, dropout=dropout),
            )

        # Apply LoRA to language model head (this one is actually nn.Linear)
        if hasattr(self.model, "lm_head"):
            # Check if it's Linear or Conv1D
            is_linear = isinstance(self.model.lm_head, nn.Linear)
            try:
                from transformers.pytorch_utils import Conv1D

                is_conv1d = isinstance(self.model.lm_head, Conv1D)
            except ImportError:
                is_conv1d = False

            if is_linear or is_conv1d:
                self.model.lm_head = apply_lora_to_linear(
                    self.model.lm_head, rank=rank, alpha=alpha, dropout=dropout
                )

        lora_applied = len(modules_to_wrap) + (
            1 if hasattr(self.model, "lm_head") else 0
        )
        print(f"   Applied LoRA to {lora_applied} layers")

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass through GPT-2 model.

        Compatible interface with BigramLanguageModel:
        - Uses 'idx' instead of 'input_ids' for consistency
        - Uses 'targets' instead of 'labels' for consistency

        Args:
            idx: Token IDs of shape (B, T) - input token indices
            targets: Optional target token IDs of shape (B, T) for loss computation

        Returns:
            logits: Model predictions of shape (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, None otherwise
        """
        # GPT-2 expects labels to be the same as input_ids for next-token prediction
        # We shift targets to align with next-token prediction
        if targets is not None:
            outputs = self.model(input_ids=idx, labels=targets)
        else:
            outputs = self.model(input_ids=idx)
        logits = outputs.logits
        loss = outputs.loss if targets is not None else None
        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text using GPT-2.

        Args:
            input_ids: Starting token IDs of shape (B, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (only sample from top k tokens)
            do_sample: If True, use sampling; if False, use greedy decoding

        Returns:
            Generated token IDs of shape (B, T + max_new_tokens)
        """
        self.model.eval()
        with torch.no_grad():
            # Create attention mask: 1 for all real tokens, 0 for padding
            # Since pad_token_id == eos_token_id, we need to explicitly set this
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.model.train()
        return generated

    def encode(self, text: str, max_length: Optional[int] = None) -> list[int]:
        """
        Encode text to token IDs using GPT-2 tokenizer.

        For long texts, chunks the text to avoid warnings about exceeding
        max_position_embeddings. The chunks are encoded separately and concatenated.

        Args:
            text: Text to encode
            max_length: Maximum sequence length per chunk (defaults to model's max_position_embeddings)

        Returns:
            List of token IDs (may be longer than max_length if text is very long)
        """
        if max_length is None:
            max_length = self.model.config.max_position_embeddings

        # For short texts, encode directly (this won't trigger warnings)
        # Estimate: GPT-2 tokenizer typically produces ~0.75 tokens per character for English
        # So for max_length=1024, we'd expect ~1365 chars to be safe
        estimated_safe_length = int(max_length * 1.3)  # 30% buffer

        if len(text) <= estimated_safe_length:
            # Short text - encode directly
            return self.tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)

        # Long text - chunk to avoid warnings
        # Use very conservative chunk size: aim for ~70% of max_length tokens per chunk
        # This gives us buffer to avoid warnings
        target_tokens_per_chunk = int(max_length * 0.7)
        # Estimate chars per token (conservative: ~4 chars per token for English)
        # But be even more conservative to account for variable tokenization
        chunk_size_chars = target_tokens_per_chunk * 3  # More conservative: 3 chars per token

        all_tokens = []
        i = 0
        while i < len(text):
            chunk = text[i : i + chunk_size_chars]
            # Encode with truncation to ensure we never exceed max_length
            chunk_tokens = self.tokenizer.encode(
                chunk, 
                add_special_tokens=False, 
                max_length=max_length, 
                truncation=True
            )
            all_tokens.extend(chunk_tokens)
            
            # Move to next chunk
            i += chunk_size_chars
            
            # If the chunk was truncated, we need to be more careful
            # But since we're using truncation=True, we're safe

        return all_tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text using GPT-2 tokenizer."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_parameter_info(self) -> dict:
        """
        Get parameter statistics (useful for LoRA).

        Returns:
            Dictionary with parameter counts and percentages
        """
        if self.use_lora and LORA_AVAILABLE:
            info = count_lora_parameters(self.model)
            # Add model name info
            info["model_name"] = self.model_name
            # Calculate LoRA percentage if not already present
            if "lora_percentage" not in info:
                info["lora_percentage"] = (
                    (info["trainable"] / info["total"] * 100)
                    if info["total"] > 0
                    else 0.0
                )
            return info
        else:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            return {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
                "lora_percentage": 0.0,
                "lora_only_percentage": 0.0,
                "model_name": self.model_name,
            }

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
