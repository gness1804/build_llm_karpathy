# Recommended Training Command - Optimized for Loss Plateau

## Changes Made to `training.py`

1. **Increased LoRA Capacity**:
   - Default rank: 8 → 16 (2x capacity)
   - Default alpha: 16 → 32 (matches rank*2)
   - More parameters = better adaptation to your data

2. **Increased Learning Rate**:
   - Default: 1e-5 → 2e-5 (2x faster convergence)
   - Still conservative for fine-tuning

3. **Added Gentle Warmup**:
   - 2% of training steps (100 steps for 5000 total)
   - Starts at 10% LR, ramps to full LR
   - Prevents early instability
   - Can be disabled with `USE_LR_WARMUP=False`

4. **Less Aggressive Gradient Clipping**:
   - LoRA: 1.0 → 2.0 (allows adapters to learn better)
   - Full fine-tuning: still 1.0

5. **Adjusted Default Batch Size**:
   - Default: 32 → 16 (matches your OOM constraint)

## Recommended Training Command

```bash
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
MODEL_TYPE=gpt2 \
USE_LORA=True \
LORA_RANK=16 \
LORA_ALPHA=32.0 \
TRAINING_STEPS=5000 \
LEARNING_RATE=2e-5 \
BLOCK_SIZE=128 \
BATCH_SIZE=16 \
ENABLE_CHECKPOINTS=True \
CHECKPOINT_INTERVAL=500 \
python3 training.py
```

**Note**: The defaults are now optimized, so you can also just use:

```bash
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
MODEL_TYPE=gpt2 \
USE_LORA=True \
BATCH_SIZE=16 \
ENABLE_CHECKPOINTS=True \
CHECKPOINT_INTERVAL=500 \
python3 training.py
```

## What to Expect

1. **Better Convergence**: Higher LR + more LoRA capacity should help loss decrease
2. **Stability**: Warmup prevents early loss spikes
3. **Monitoring**: Watch for loss increases at checkpoints (red flag)

## If Loss Still Plateaus

### Option A: Try Even Higher LoRA Capacity
```bash
LORA_RANK=32 \
LORA_ALPHA=64.0 \
# ... rest of command
```

### Option B: Disable Warmup (if it's causing issues)
```bash
USE_LR_WARMUP=False \
# ... rest of command
```

### Option C: Try Full Fine-Tuning (if memory allows)
```bash
USE_LORA=False \
BATCH_SIZE=8 \
BLOCK_SIZE=256 \
LEARNING_RATE=1e-5 \
# ... rest of command
```

### Option D: Increase Block Size (if memory allows)
```bash
BLOCK_SIZE=256 \
BATCH_SIZE=8 \
# ... rest of command
```

## Monitoring Checklist

- [ ] Loss decreases at each checkpoint (not increases!)
- [ ] Train and val loss both decreasing (not diverging)
- [ ] Loss improvement > 0.1 per 500 steps (good progress)
- [ ] No loss spikes > 1.0 (indicates instability)

## Expected Loss Progression

For a 5MB dataset with GPT-2:
- **Initial**: ~6.5-7.0 (GPT-2 base loss on your data)
- **After 1000 steps**: Should see ~0.2-0.5 improvement
- **After 5000 steps**: Should see ~0.5-1.0 improvement
- **Target**: < 5.5 for good quality generation

If loss doesn't improve by at least 0.3 after 2000 steps, try the alternatives above.









