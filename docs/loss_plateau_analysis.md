# Loss Plateau Diagnostic Report

## Executive Summary
Your GPT-2 fine-tuning exhibits a **severe loss plateau with a catastrophic spike at step 2000**, followed by inability to recover. The model shows good initial progress (9.23 → 6.65 loss over 1500 steps) but then stalls.

---

## Key Observations from Your Logs

### 1. Loss Progression Timeline

| Step | Train Loss | Val Loss | Δ Train | Status |
|------|-----------|----------|---------|--------|
| 0 | 9.2270 | 9.2680 | - | Baseline |
| 500 | 6.9544 | 7.1291 | -2.273 | ✅ Good progress |
| 1000 | 6.5811 | 6.7096 | -0.373 | ✅ Continuing descent |
| 1500 | 6.6511 | 6.7427 | +0.070 | ⚠️ Slight degradation |
| **2000** | **9.5846** | **9.6169** | **+2.932** | **🔴 CATASTROPHIC SPIKE** |
| 2500 | 6.6424 | 6.7105 | -2.942 | ⚠️ Recovered but stuck |

### 2. The Catastrophic Spike (Step 1500 → 2000)
- **Nature**: Loss jumped from 6.65 → 9.58 (40% increase)
- **Magnitude**: 3.0 loss points in a single eval interval (500 steps)
- **Recovery**: Partial recovery to 6.64 by step 2500, but now stuck in plateau
- **Pattern**: Suggests **gradient explosion** or **problematic data batch**

### 3. Training Configuration at Spike Time

```
Batch size: 8 (very small - noisy gradients)
Block size: 256 (large - computes loss on 256 tokens at once)
Learning rate: 2e-05 (with warmup schedule)
LR schedule at step 2000: LinearWarmup(500 steps) + CosineDecay
```

**Problem Identified**: At step 2000 (which is 500 steps into decay phase), the learning rate was descending via cosine schedule, but this coincided with possible data issues.

### 4. Training Speed Degradation

- **Steps 0-500**: 1.04 steps/sec
- **Steps 500-1500**: 1.05 steps/sec  
- **Steps 1500-2000**: 1.05 steps/sec (before spike)
- **Steps 2000-2500**: 0.62 steps/sec (dropped 40% after spike - MPS memory thrashing?)
- **Steps 2500+**: Continues slow ~0.6 steps/sec

**Hypothesis**: The spike consumed abnormal memory/gradients, leaving MPS in a bad state. Block size of 256 with batch size 8 creates massive attention matrices.

### 5. Divergence Pattern

- **Pre-spike (step 0-1500)**: Train ≈ Val (both descending together)
- **Post-spike (step 1500+)**: Train > Val consistently (7% higher)
- **Interpretation**: Model is overfitting or data distribution shifted

---

## Root Cause Analysis

### Why Loss Plateaued:

1. **Gradient Explosion at Step 2000**
   - Warmup + cosine schedule interaction may have caused instability
   - Block size 256 with batch 8 = sparse, noisy gradients with extreme attention patterns
   - No gradient clipping recorded in your hyperparameters (but it IS in code with max_norm=1.0)

2. **Data Quality Issues**
   - The training data (`training_data_final_merged.md` - 5.0 MB) is Q&A format
   - At ~2000 steps with batch_size=8 and block_size=256, you've seen ~2000×8×256 = 4M tokens
   - If there's a repetitive section or distribution shift in the data, it would hit around this point

3. **Small Batch Size**
   - Batch size 8 with 4 attention heads means each head sees only 2 examples
   - This causes very noisy gradients and instability with LoRA

4. **Suboptimal Learning Rate Schedule**
   - Warmup: 0.1 → 1.0 over 500 steps (ramping up to 2e-05)
   - Decay: Cosine annealing for 4500 steps
   - At step 2000: LR ≈ 1.55e-05 (still on decay)
   - This schedule is **very gentle** and doesn't allow aggressive learning

5. **Block Size Causing MPS Performance Cliff**
   - Block size 256 requires enormous attention computation
   - This likely triggered MPS memory issues, causing training to slow to 0.6 steps/sec
   - MPS garbage collection may have left bad state

---

## Why You're Stuck at ~6.6 Loss:

1. **Model is fit for frequency statistics of your data** (Q&A format)
2. **Learning rate too low to escape local minimum** (2e-05 is very conservative)
3. **Batch size too small for stable learning** (8 examples per batch)
4. **No learning rate warmup after spike** (stays at conservative value)

---

## Recommended Fixes (Priority Order)

### Immediate (Next Training Run)

1. **Reduce block size from 256 → 128**
   - Restore MPS speed (target: 1.0 steps/sec)
   - Reduces attention computation by 4x
   - Should recover the 40% speed loss

2. **Increase batch size from 8 → 32**
   - Stabilize gradients (4x more stable)
   - Better signal-to-noise ratio
   - Use gradient accumulation if OOM: `effective_batch = 32` = `4 accumulation steps × batch 8`

3. **Simplify learning rate schedule**
   - Drop warmup/cosine schedule initially
   - Use constant LR = 1e-5 for diagnostic run
   - If that doesn't help, try **1e-4** (10x higher)

4. **Resume from step 1500 checkpoint**
   - This is your last good checkpoint before the spike
   - Step 2000 and later are corrupted by the spike

### Longer Term

5. **Add gradient logging**
   - Log max grad norm every 100 steps
   - Detect future spikes before they crash training

6. **Data analysis**
   - Check for repetitions/duplicates around token positions 2M-2.5M
   - Consider shuffling data with seed for reproducibility

7. **Switch to mixed precision training**
   - Try `torch.autocast("mps")` to reduce precision and improve MPS stability
   - May reduce gradient noise and spike risk

---

## Testing Your Hypothesis

Run this quick test:
```bash
TRAINING_STEPS=2000 \
BLOCK_SIZE=128 \
BATCH_SIZE=32 \
LEARNING_RATE=1e-5 \
python training.py
```

**Expected outcome**: Should plateau at ~6.2-6.4 loss (better than current 6.6)

---

## Files of Interest

- **Best checkpoint**: `checkpoints/checkpoint_gpt2_training_data_final_merged_step001500_11222025_112739.pt` (step 1500, loss 6.65)
- **Corrupted checkpoints**: Steps 2000+ should be ignored
- **Training data**: `sources/training_data_final_merged.md` (5.0 MB, Q&A format)

