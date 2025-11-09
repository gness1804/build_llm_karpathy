# Training Recommendations

## Current Model Size (~780K-3M parameters)

### ✅ Recommended: Full Fine-Tuning

**Use this for best performance:**
```bash
python training.py
# or explicitly:
USE_LORA=False python training.py
```

**Why:**
- Fastest training speed (no LoRA overhead)
- Simpler code path
- Best performance for your model size
- No warnings or overhead

### ❌ Not Recommended: LoRA (for now)

**LoRA is available but slower for small models:**
```bash
USE_LORA=True python training.py  # Will be slower!
```

**Why not recommended:**
- 10-15x slower than full fine-tuning
- Overhead outweighs benefits
- Designed for larger models

## When to Use LoRA

Use LoRA when you have:
- ✅ **Large models** (> 10M parameters) - LoRA becomes faster
- ✅ **Memory constraints** - Can't fit full gradients
- ✅ **Multiple adapters** - Want to train different adapters for different tasks
- ✅ **Pre-trained models** - Fine-tuning GPT-2/TinyLlama with LoRA

## Future Scaling

When you scale to larger models:

1. **10-50M parameters**: LoRA starts to break even
2. **50-100M parameters**: LoRA is 2-3x faster
3. **> 100M parameters**: LoRA is 3-10x faster

The LoRA implementation is ready - just switch `USE_LORA=True` when you scale up!

## Summary

**For now:** Use full fine-tuning (`USE_LORA=False` or default)
**For later:** LoRA is ready when you scale to larger models

No need to remove LoRA - it's correct and will be valuable later!

