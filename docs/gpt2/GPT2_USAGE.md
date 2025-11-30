# GPT-2 Integration Usage Guide

## Overview

The training script now supports fine-tuning pre-trained GPT-2 models from HuggingFace. This provides transfer learning benefits - start with a model that already understands language, then fine-tune on your specific data.

## Quick Start

### Use GPT-2 (Recommended for Better Results)

```bash
# Basic GPT-2 fine-tuning (GPT-2 Small, 124M params)
MODEL_TYPE=gpt2 python training.py

# GPT-2 with LoRA (most efficient)
MODEL_TYPE=gpt2 USE_LORA=True python training.py

# Larger GPT-2 model
MODEL_TYPE=gpt2 GPT2_MODEL_NAME=gpt2-medium USE_LORA=True python training.py
```

### Use From-Scratch Model (Original)

```bash
# Original from-scratch training
python training.py
# or explicitly:
MODEL_TYPE=from_scratch python training.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | `from_scratch` | Model type: `from_scratch` or `gpt2` |
| `GPT2_MODEL_NAME` | `gpt2` | GPT-2 model size: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| `USE_LORA` | `False` | Enable LoRA fine-tuning (recommended for GPT-2) |
| `LORA_RANK` | `8` | LoRA rank (if USE_LORA=True) |
| `LORA_ALPHA` | `16.0` | LoRA alpha (if USE_LORA=True) |

## GPT-2 Model Sizes

| Model | Parameters | Memory | Speed | Quality |
|-------|-----------|--------|-------|---------|
| `gpt2` | 124M | ~500MB | Fast | Good |
| `gpt2-medium` | 355M | ~1.4GB | Medium | Better |
| `gpt2-large` | 774M | ~3GB | Slow | Great |
| `gpt2-xl` | 1.5B | ~6GB | Very Slow | Excellent |

**Recommendation**: Start with `gpt2` (Small) + LoRA for best balance.

## Examples

### Example 1: GPT-2 Small with LoRA (Recommended)

```bash
MODEL_TYPE=gpt2 \
USE_LORA=True \
LORA_RANK=8 \
python training.py
```

**Benefits:**
- Pre-trained language knowledge
- Only trains ~0.1-1% of parameters (LoRA)
- Fast training
- Better results than from-scratch

### Example 2: GPT-2 Medium Full Fine-Tuning

```bash
MODEL_TYPE=gpt2 \
GPT2_MODEL_NAME=gpt2-medium \
USE_LORA=False \
python training.py
```

**Note**: Full fine-tuning trains all parameters. Use LoRA for efficiency.

### Example 3: Quick Test with GPT-2

```bash
MODEL_TYPE=gpt2 \
USE_LORA=True \
TEST_MODE=True \
python training.py
```

## Why GPT-2 + LoRA is Recommended

1. **Pre-trained Knowledge**: GPT-2 already understands language
2. **Faster Convergence**: 100-1000x fewer training steps needed
3. **LoRA Efficiency**: Only train 0.1-1% of parameters
4. **Better Results**: Outperforms from-scratch training
5. **Cost Effective**: Fast training, low memory usage

## Comparison

| Approach | Training Time | Parameters Trained | Quality | Cost |
|----------|--------------|-------------------|---------|------|
| From-scratch | 100% | 100% | Baseline | $0 (local) |
| GPT-2 Full FT | 50-80% | 100% | Better | $0 (local) |
| GPT-2 + LoRA | 10-30% | 0.1-1% | Best | $0 (local) |

## Tokenization

GPT-2 uses its own BPE tokenizer (50,257 tokens). The training script automatically:
- Loads GPT-2's tokenizer when `MODEL_TYPE=gpt2`
- Encodes your training data with GPT-2 tokenizer
- Handles tokenization transparently

**Note**: You don't need to set `TOKENIZATION_METHOD` when using GPT-2 - it uses GPT-2's tokenizer automatically.

## Output Files

Output filenames now include model type:
- GPT-2: `build_llm_output_gpt2_..._gpt2_gpt2_lora_r8_a16.0_...`
- From-scratch: `build_llm_output_bigram_..._from_scratch_...`

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Solution**: Install transformers:
```bash
pip install transformers>=4.35.0
```

### Issue: "Out of memory" with GPT-2 Large/XL
**Solution**: 
- Use smaller model: `GPT2_MODEL_NAME=gpt2`
- Use LoRA: `USE_LORA=True`
- Reduce batch size in training script

### Issue: Slow training with GPT-2
**Solution**: 
- Use LoRA: `USE_LORA=True` (trains 90-99% fewer parameters)
- Use smaller model: `GPT2_MODEL_NAME=gpt2`
- Use test mode: `TEST_MODE=True` for quick testing

## Next Steps

1. **Try GPT-2 Small + LoRA**: Best starting point
2. **Compare Results**: Run from-scratch vs GPT-2 and compare outputs
3. **Experiment**: Try different GPT-2 sizes and LoRA ranks
4. **Scale Up**: When ready, try GPT-2 Medium or Large

## Technical Details

- GPT-2 models are loaded from HuggingFace automatically
- First run downloads the model (~500MB for GPT-2 Small)
- Models are cached locally for future use
- LoRA is applied to attention and feed-forward layers
- Compatible with existing training loop and evaluation

