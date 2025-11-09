# LoRA Fine-Tuning Usage Guide

## Overview

LoRA (Low-Rank Adaptation) allows you to fine-tune your language model efficiently by only training a small fraction (typically 0.1-1%) of the model parameters.

### ⚠️ Important Note for Small Models

**For models < 5M parameters, LoRA may be SLOWER than full fine-tuning!**

LoRA is designed for large models where gradient computation is the bottleneck. For small models, the LoRA overhead (extra forward passes, wrapper calls) can outweigh the benefits.

**Recommendation for current model size (~780K-3M params):**
- Use **full fine-tuning** (`USE_LORA=False`) for best speed
- LoRA will become beneficial when you scale to larger models (>10M params)

### Benefits (for larger models)

- **90-99% fewer trainable parameters**
- **3-10x faster training** (for models >10M params)
- **Lower memory usage**
- **Same or better results**

## Quick Start

### Recommended: Full Fine-Tuning (for current model size)

```bash
# Train all parameters (fastest for small models)
python training.py

# Or explicitly disable LoRA
USE_LORA=False python training.py
```

### Enable LoRA Training (for larger models)

**Note:** Only use LoRA if you have a model > 10M parameters, or if you need to train multiple adapters.

```bash
# Basic LoRA training (uses default rank=8, alpha=16.0)
USE_LORA=True python training.py

# Custom LoRA parameters
USE_LORA=True LORA_RANK=16 LORA_ALPHA=32.0 python training.py

# With test mode for quick testing
USE_LORA=True TEST_MODE=True python training.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LORA` | `False` | Enable LoRA fine-tuning (`True`/`False`) |
| `LORA_RANK` | `8` | Rank of LoRA matrices (typically 4-32) |
| `LORA_ALPHA` | `16.0` | LoRA scaling factor (typically rank * 2) |
| `LORA_DROPOUT` | `0.0` | Dropout probability for LoRA path |

## LoRA Parameters Explained

### Rank (`LORA_RANK`)
- **Lower rank** (4-8): Fewer parameters, faster training, may have slightly lower quality
- **Higher rank** (16-32): More parameters, better quality, still much faster than full fine-tuning
- **Recommended**: Start with 8, increase to 16 if needed

### Alpha (`LORA_ALPHA`)
- Controls the scaling of LoRA contributions
- **Typical value**: `rank * 2` (e.g., rank=8 → alpha=16)
- **Higher alpha**: Stronger LoRA influence
- **Lower alpha**: Weaker LoRA influence

### Dropout (`LORA_DROPOUT`)
- Regularization for LoRA adapters
- **Typical value**: 0.0-0.1
- Helps prevent overfitting

## Examples

### Example 1: Quick Test with LoRA
```bash
USE_LORA=True TEST_MODE=True python training.py
```

### Example 2: Production LoRA Training
```bash
USE_LORA=True \
LORA_RANK=16 \
LORA_ALPHA=32.0 \
LORA_DROPOUT=0.1 \
python training.py
```

### Example 3: Compare LoRA vs Full Fine-Tuning
```bash
# Full fine-tuning
python training.py > output_full.txt

# LoRA fine-tuning
USE_LORA=True python training.py > output_lora.txt

# Compare parameter counts and training times
```

## What Gets Trained?

### With LoRA Enabled:
- ✅ **LoRA adapters** (0.1-1% of parameters) - Trainable
- ❌ **Base model weights** - Frozen
- ✅ **Embedding layers** - Trainable (always)

### Without LoRA:
- ✅ **All parameters** - Trainable

## Parameter Savings

When you run training with LoRA, you'll see output like:

```
🔧 Using LoRA for efficient fine-tuning
   LoRA rank: 8, alpha: 16.0, dropout: 0.0
📊 Parameter Statistics:
   Total parameters: 1,234,567
   Trainable (LoRA): 12,345
   Frozen (base): 1,222,222
   LoRA percentage: 1.00%
   💰 Cost savings: Training only 1.00% of parameters!
```

This shows you're only training ~1% of parameters while still getting the full model's capabilities!

## Architecture Details

LoRA is applied to:
- **Attention layers**: key, query, value projections, and output projection
- **Feed-forward layers**: both linear layers in the MLP
- **Language model head**: final output projection

The base weights remain frozen, so you can:
- Train multiple LoRA adapters for different tasks
- Switch between adapters without retraining
- Combine multiple adapters (future feature)

## Performance Comparison

| Method | Trainable Params | Training Time | Memory Usage |
|--------|-----------------|---------------|--------------|
| Full Fine-Tuning | 100% | 100% | 100% |
| LoRA (rank=8) | ~1% | ~30% | ~50% |
| LoRA (rank=16) | ~2% | ~40% | ~60% |

*Times are approximate and depend on your hardware*

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'lora'"
**Solution**: Make sure you're in the project root directory when running training.py

### Issue: Training is slower than expected
**Solution**: 
- Check that LoRA is actually enabled (`USE_LORA=True`)
- Verify parameter counts show low trainable percentage
- Ensure optimizer is only training LoRA parameters

### Issue: Model quality is lower with LoRA
**Solution**:
- Increase `LORA_RANK` (try 16 or 32)
- Increase `LORA_ALPHA` (try rank * 2)
- Add dropout (`LORA_DROPOUT=0.1`)

## Advanced Usage

### Loading a LoRA Model

```python
from models.bigram_lm_v2_lora import BigramLanguageModelLoRA

# Create model with LoRA
model = BigramLanguageModelLoRA(
    vocab_size=vocab_size,
    n_embd=256,
    block_size=128,
    device='mps',
    dropout=0.2,
    n_head=4,
    n_layer=4,
    use_lora=True,
    lora_rank=8,
    lora_alpha=16.0,
)

# Get parameter info
info = model.get_parameter_info()
print(f"Trainable: {info['trainable']:,} ({info['lora_percentage']:.2f}%)")
```

### Saving and Loading LoRA Weights

```python
# Save only LoRA weights (much smaller file!)
lora_state = {
    name: param for name, param in model.named_parameters()
    if 'lora' in name and param.requires_grad
}
torch.save(lora_state, 'lora_weights.pt')

# Load LoRA weights
lora_state = torch.load('lora_weights.pt')
model.load_state_dict(lora_state, strict=False)
```

## Next Steps

1. **Try LoRA with different ranks**: Compare rank=4, 8, 16, 32
2. **Experiment with alpha**: Try alpha = rank, rank*2, rank*4
3. **Add dropout**: Test with LORA_DROPOUT=0.1 for regularization
4. **Compare results**: Run full fine-tuning vs LoRA and compare outputs

## References

- Original LoRA paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- HuggingFace PEFT library: [Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)

