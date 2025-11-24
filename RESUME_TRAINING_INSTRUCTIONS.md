# Resuming Training with Diagnostic Hyperparameters

## Summary of Changes

Your training script has been updated with the recommended diagnostic hyperparameters from the loss plateau analysis:

### Changes Made to `training.py`:

1. **Reduced block size**: 256 → 128
2. **Increased batch size**: 8 → 32
3. **Learning rate schedule**: Changed from warmup + cosine decay → constant (1e-5)

These changes address the root causes identified in the plateau analysis:
- Smaller block size reduces attention computation overhead and improves MPS performance
- Larger batch size stabilizes gradients and reduces noisy updates
- Constant learning rate eliminates schedule instability that triggered the spike at step 2000

---

## How to Resume Training

Use the existing `training_resume.py` script with the best checkpoint (step 1500):

### Command

```bash
CHECKPOINT_PATH=checkpoints/checkpoint_gpt2_training_data_final_merged_step001500_11222025_112739.pt \
BATCH_SIZE=32 \
BLOCK_SIZE=128 \
LEARNING_RATE=1e-5 \
RESUME_STEPS=2000 \
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
python3 training_resume.py
```

### Environment Variables Explained

| Variable | Value | Notes |
|----------|-------|-------|
| `CHECKPOINT_PATH` | `checkpoints/checkpoint_gpt2_training_data_final_merged_step001500_...` | Best checkpoint before the spike (step 1500, loss 6.65) |
| `BATCH_SIZE` | `32` | Increased from 8 for gradient stability (4x more stable) |
| `BLOCK_SIZE` | `128` | Reduced from 256 (4x faster attention computation) |
| `LEARNING_RATE` | `1e-5` | Constant LR (no warmup/cosine decay) |
| `RESUME_STEPS` | `2000` | Train for 2000 additional steps (total: 1500 + 2000 = 3500) |
| `TRAINING_DATA_SOURCE` | `sources/training_data_final_merged.md` | Same training data |

---

## What to Expect

### Performance Targets

- **Training speed**: Should recover to ~1.0 steps/sec (from degraded 0.6 steps/sec)
- **Loss trajectory**: Should reach 6.2-6.4 validation loss (improvement from 6.6)
- **No spikes**: Training should be stable with no sudden loss jumps

### Timeline

- **Duration**: ~2000 steps × ~1.0 steps/sec = ~33 minutes
- **Checkpoints**: Saved every 500 steps
- **Output**: Training log saved to `logs/` and output summary to `outputs/`

---

## Monitoring Training

Check the loss progression by looking at checkpoint output:

```bash
tail -f logs/resume_training_log_*.log
```

Expected loss progression:
```
Step 1500: train loss ~6.65, val loss ~6.74 (starting point)
Step 2000: train loss ~6.5, val loss ~6.65 (improving)
Step 2500: train loss ~6.3, val loss ~6.5 (good progress)
Step 3000: train loss ~6.2, val loss ~6.4 (target reached)
```

---

## If Loss Still Plateaus

If the diagnostic run shows no improvement, try:

1. **Increase learning rate** to `5e-5` or `1e-4`
   ```bash
   LEARNING_RATE=5e-5 python3 training_resume.py
   ```

2. **Reduce block size further** to `64`
   ```bash
   BLOCK_SIZE=64 BATCH_SIZE=32 python3 training_resume.py
   ```

3. **Check data quality** - there may be anomalies around token position 2M-2.5M
   ```bash
   python3 sources/scripts/clean_training_data.py --input sources/training_data_final_merged.md
   ```

---

## Checkpoint Details

### Current State (Step 1500)
- **Location**: `checkpoints/checkpoint_gpt2_training_data_final_merged_step001500_11222025_112739.pt`
- **Train loss**: 6.6511
- **Val loss**: 6.7427
- **Status**: Last good checkpoint before spike

### Avoid These Checkpoints
- Step 2000+: Corrupted by the catastrophic spike
- These show artificially high loss values (9.58)

---

## Next Steps After Diagnostic Run

Once you verify that the new hyperparameters improve training:

1. **Run full training**: Extend to 5000+ steps
2. **Re-enable learning rate schedule**: Cosine annealing after initial stability
3. **Add gradient monitoring**: Log gradient norms to detect future instability
4. **Data analysis**: Investigate token positions around 2M mark for anomalies

---

## Reference

See `loss_plateau_analysis.md` for the complete diagnostic report and root cause analysis.
