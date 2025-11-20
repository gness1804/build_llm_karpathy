# Training Improvements for GPT-2 Fine-tuning

## Current Issues
- **Loss: 5-6** (very high - GPT-2 should be ~2-3 for coherent output)
- **Incoherent output** despite pre-trained model
- **700KB dataset** trained for 5,000 steps

## Current Configuration
- Learning rate: `3e-4` (0.0003) - **TOO HIGH for fine-tuning**
- Block size: `64` - **TOO SMALL** (GPT-2 trained on 1024)
- Batch size: `16`
- Optimizer: AdamW
- No learning rate schedule (warmup/decay)
- Using LoRA

## Key Problems & Solutions

### 1. Learning Rate Too High ⚠️ **CRITICAL**
**Problem**: `3e-4` is appropriate for training from scratch, but **way too high** for fine-tuning a pre-trained model.

**Solution**: Use **1e-5 to 5e-5** (0.00001 to 0.00005) for fine-tuning
- Pre-trained models need gentle updates to preserve knowledge
- High learning rates cause "catastrophic forgetting" → gibberish output

**Fix in training.py**:
```python
# For GPT-2 fine-tuning, use much lower learning rate
if MODEL_TYPE == "gpt2":
    learning_rate = 1e-5  # or 5e-5 for slightly faster convergence
else:
    learning_rate = 3e-4  # OK for from-scratch training
```

### 2. Block Size Too Small ⚠️ **IMPORTANT**
**Problem**: `block_size = 64` is very small. GPT-2 was trained on sequences of 1024 tokens.

**Solution**: Increase to at least **256-512** (or 1024 if memory allows)
- Larger context = better understanding of relationships
- GPT-2's attention mechanism works better with longer sequences

**Fix**:
```python
if MODEL_TYPE == "gpt2":
    block_size = 512  # or 1024 if you have memory
    batch_size = 8    # Reduce batch size to compensate for larger block_size
```

### 3. No Learning Rate Schedule ⚠️ **IMPORTANT**
**Problem**: Constant learning rate doesn't adapt during training.

**Solution**: Add learning rate warmup + decay
- Warmup: Start low, gradually increase (prevents early instability)
- Decay: Gradually decrease (helps convergence)

**Fix** - Add to training.py:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# After optimizer initialization
if MODEL_TYPE == "gpt2":
    # Warmup for first 10% of steps
    warmup_steps = int(0.1 * training_steps)
    # Cosine decay for remaining 90%
    decay_steps = training_steps - warmup_steps
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, decay_scheduler], milestones=[warmup_steps])
else:
    scheduler = None
```

Then in training loop, after `optimizer.step()`:
```python
if scheduler is not None:
    scheduler.step()
```

### 4. More Training Steps Needed
**Problem**: 5,000 steps might not be enough for 700KB dataset.

**Solution**: 
- Monitor validation loss - train until it plateaus or starts increasing
- Consider 10,000-20,000 steps for better convergence
- Use early stopping based on validation loss

### 5. LoRA Configuration
**Current**: Using LoRA (good for efficiency)

**Recommendations**:
- LoRA rank: 8 (current) is fine, but try 16 for better capacity
- LoRA alpha: 16 (current) is good (2x rank)
- Consider training embeddings too if dataset is domain-specific

### 6. Data Preprocessing
**Check**:
- Are QUESTION/ANSWER labels properly formatted?
- Is there enough variety in the data?
- Consider data augmentation (if appropriate)

### 7. Gradient Clipping
**Add gradient clipping** to prevent exploding gradients:
```python
# After loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Recommended Training Configuration

```python
# For GPT-2 fine-tuning
MODEL_TYPE = "gpt2"
USE_LORA = True
LORA_RANK = 16  # Increased from 8
LORA_ALPHA = 32  # 2x rank

# Hyperparameters
learning_rate = 1e-5  # Much lower for fine-tuning
block_size = 512      # Increased from 64
batch_size = 8        # Reduced to fit larger block_size
training_steps = 10000  # More steps
eval_interval = 500

# Add learning rate schedule
# Add gradient clipping
```

## Expected Results
With these changes:
- **Loss should drop to 2-3** (from 5-6)
- **Output should be coherent** (not gibberish)
- **Training will be slower** but more stable
- **Better preservation of GPT-2's knowledge**

## Training Strategy
1. **Start conservative**: LR=1e-5, block_size=256, 10K steps
2. **Monitor validation loss**: Should decrease steadily
3. **If loss plateaus**: Try slightly higher LR (2e-5) or more steps
4. **If loss increases**: LR too high, reduce to 5e-6
5. **If still gibberish**: Check data quality, increase block_size

## Additional Tips from Research
- **Small datasets** (like 700KB) need **lower learning rates** and **more careful training**
- **Pre-trained models** are sensitive - small changes can break coherence
- **Validation loss** is more important than training loss for quality
- **Temperature during generation** (0.7-0.9) can help coherence

## Quick Fix Priority
1. **Lower learning rate to 1e-5** (highest impact)
2. **Increase block_size to 512** (high impact)
3. **Add learning rate schedule** (medium impact)
4. **Add gradient clipping** (safety)
5. **Increase training steps** (if needed)

