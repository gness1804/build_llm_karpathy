# Training Optimization Plan - Loss Plateau Analysis

## Current Issues Observed

From the log file (`resume_training_log_gpt2_training_data_final_merged_11222025_150119.log`):

1. **Loss Increased at Step 2000**: 
   - Train: 6.67 → 6.75 (worse)
   - Val: 6.71 → 6.93 (much worse)
   - This indicates training instability

2. **Loss Plateauing**: 
   - Step 1500: train 6.67, val 6.71
   - Step 2500: train 6.52, val 6.72
   - Minimal improvement over 1000 steps

3. **Current Configuration**:
   - Batch size: 16 (reduced from 32 due to OOM)
   - Block size: 128
   - Learning rate: 1e-5 (constant, no scheduler)
   - LoRA rank: 8 (default)
   - LoRA alpha: 16.0 (default)
   - Gradient clipping: 1.0

## Root Causes

### 1. **LoRA Capacity Too Low**
- Rank 8 with alpha 16 may not have enough capacity for the task
- GPT-2 (124M params) fine-tuning often needs rank 16-32

### 2. **Learning Rate Issues**
- 1e-5 might be too low, causing slow convergence
- Constant LR without warmup can cause early instability
- No decay means model can't fine-tune as it learns

### 3. **Batch Size Instability**
- Reduced from 32 to 16 due to OOM
- Smaller batches = noisier gradients = instability
- Need to compensate with other techniques

### 4. **Block Size Too Small**
- 128 tokens may not capture enough context for Q&A pairs
- GPT-2 was trained on 1024 tokens, using 128 is a big reduction

### 5. **Gradient Clipping Too Aggressive**
- Clipping at 1.0 might be preventing necessary gradient updates
- LoRA adapters need gradients to flow properly

## Recommended Solutions (Prioritized)

### Option 1: Increase LoRA Capacity (Easiest, Low Risk)
**Why**: More parameters = better adaptation to your data

```bash
USE_LORA=True \
LORA_RANK=16 \
LORA_ALPHA=32.0 \
BATCH_SIZE=16 \
BLOCK_SIZE=128 \
LEARNING_RATE=2e-5 \
python training.py
```

**Expected**: Better convergence, more stable training

### Option 2: Add Conservative Learning Rate Schedule (Medium Risk)
**Why**: Warmup prevents early instability, decay helps convergence

Modify `training.py` to add a gentle warmup + decay:
- 5% warmup (very short)
- Linear decay to 50% of initial LR

### Option 3: Increase Block Size (If Memory Allows)
**Why**: More context = better understanding of Q&A structure

Try `BLOCK_SIZE=256` if you can fit it in memory with batch_size=8

### Option 4: Reduce Gradient Clipping (Low Risk)
**Why**: Less aggressive clipping allows LoRA adapters to learn better

Change from `max_norm=1.0` to `max_norm=2.0` or `max_norm=5.0`

### Option 5: Try Full Fine-Tuning (If Model Size Allows)
**Why**: For GPT-2 (124M), full fine-tuning might work better than LoRA

```bash
USE_LORA=False \
BATCH_SIZE=8 \
BLOCK_SIZE=256 \
LEARNING_RATE=1e-5 \
python training.py
```

## Recommended Training Command (Best Balance)

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

**Changes from current**:
- LoRA rank: 8 → 16 (2x capacity)
- LoRA alpha: 16 → 32 (matches rank*2)
- Learning rate: 1e-5 → 2e-5 (2x, should help convergence)

## Alternative: Conservative Learning Rate Schedule

If you want to keep constant LR but add stability, we can add a very short warmup:

```python
# 2% warmup (100 steps for 5000 total)
warmup_steps = 100
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
# Then constant for rest
```

This gives a gentle start without the complexity of decay.

## Monitoring Strategy

1. **Watch for loss increases**: If loss increases at any checkpoint, training is unstable
2. **Compare train vs val**: Large gap = overfitting, both high = underfitting
3. **Check gradient norms**: If they're always < 0.1, clipping might be too aggressive
4. **Monitor learning rate**: If using scheduler, verify it's changing as expected

## Next Steps

1. Try Option 1 first (increase LoRA rank/alpha) - lowest risk
2. If still plateauing, add gentle warmup (Option 2)
3. If memory allows, try larger block size (Option 3)
4. As last resort, try full fine-tuning (Option 5)





