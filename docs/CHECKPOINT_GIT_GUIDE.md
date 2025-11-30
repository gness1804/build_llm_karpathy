# Checkpoint Git Management Guide

This document explains how to manage model checkpoints with Git while keeping your repository lean.

## Problem

Model checkpoints are large files (1MB - 6GB+) that bloat your Git repository. By default, they should not be committed.

## Solution: Selective Checkpoint Storage

### 1. Checkpoints are Ignored by Default

The `.gitignore` file prevents automatic commits:
```
checkpoints/*.pt        # All .pt files ignored
!checkpoints/.gitkeep   # But keep directory in git
```

### 2. Keeping Important Checkpoints

You can manually add specific checkpoints you want to keep:

```bash
# Add a specific checkpoint
git add --force checkpoints/checkpoint_mymodel_step5000_11102025.pt
git commit -m "chore: add important checkpoint at step 5000"

# Or multiple checkpoints
git add --force checkpoints/checkpoint_*_step5000_*.pt
git add --force checkpoints/checkpoint_*_step10000_*.pt
git commit -m "chore: add milestone checkpoints"
```

### 3. Reducing Checkpoint Size (Optional)

Remove optimizer state to reduce size by ~50% (inference only):

```bash
# Compress a single checkpoint
python3 scripts/compress_checkpoint.py checkpoints/checkpoint_model_step5000.pt

# Compress multiple checkpoints
python3 scripts/compress_checkpoint.py "checkpoints/checkpoint_*.pt"

# Replace original (removes optimizer, saves space)
python3 scripts/compress_checkpoint.py checkpoint.pt --replace
```

This creates `checkpoint_model_step5000_compressed.pt` (~50% smaller) suitable for Git and inference.

**Trade-off**: Compressed checkpoints work for inference but can't resume training.

### 4. Workflow: Train → Compress → Commit

```bash
# Step 1: Train with checkpoints
ENABLE_CHECKPOINTS=true CHECKPOINT_INTERVAL=500 python3 training.py

# Step 2: List checkpoints
ls -lh checkpoints/ | tail -10

# Step 3: Compress important ones
python3 scripts/compress_checkpoint.py "checkpoints/checkpoint_*_step5000_*.pt"
python3 scripts/compress_checkpoint.py "checkpoints/checkpoint_*_step10000_*.pt"

# Step 4: Check file sizes
ls -lh checkpoints/ | grep compressed

# Step 5: Commit compressed versions
git add --force checkpoints/checkpoint_*_compressed.pt
git commit -m "chore: add compressed checkpoints for inference"

# Step 6: Cleanup (keep only compressed versions)
rm checkpoints/checkpoint_*.pt  # Remove original large files
git add -u  # Stage deletions if they were tracked
```

## Best Practices

### What to Commit

✅ **DO commit compressed checkpoints:**
- Final/milestone checkpoints (step 5000, 10000, etc.)
- Best-performing model based on validation loss
- Different model variants for comparison

✅ **DO commit:**
- `checkpoints/.gitkeep` (directory placeholder)
- Compression and loading scripts
- Documentation

### What NOT to Commit

❌ **DON'T commit full-size checkpoints:**
- Intermediate training checkpoints (too large)
- Multiple versions of the same model
- All training steps (keep only milestones)

### Storage Estimates

| Model | Full Checkpoint | Compressed | Savings |
|-------|-----------------|-----------|---------|
| from_scratch (small) | 2-5 MB | 1-3 MB | 40-50% |
| GPT-2 base | ~500 MB | ~250 MB | ~50% |
| GPT-2 medium | ~1.5 GB | ~750 MB | ~50% |
| GPT-2 large | ~3 GB | ~1.5 GB | ~50% |
| GPT-2 XL | ~6 GB | ~3 GB | ~50% |

### Example Commits

```bash
# Milestone checkpoints
git add --force checkpoints/*_step5000_compressed.pt
git commit -m "chore: add checkpoint at training step 5000"

# Compare two model variants
git add --force checkpoints/checkpoint_gpt2_*_compressed.pt
git commit -m "feat: add GPT-2 fine-tuned checkpoints for comparison"

# Best model found
git add --force checkpoints/checkpoint_best_val_loss_compressed.pt
git commit -m "chore: save best model checkpoint (val_loss=2.15)"
```

## Using Compressed Checkpoints

Compressed checkpoints work with `scripts/load_checkpoint.py`:

```bash
# Inference with compressed checkpoint
CHECKPOINT_PATH=checkpoints/checkpoint_model_step5000_compressed.pt \
MODE=inference \
PROMPT="Hello" \
python3 scripts/load_checkpoint.py
```

**Limitation**: Cannot resume training from compressed checkpoints (optimizer state removed).

## Undoing Accidental Commits

If you accidentally committed a large checkpoint:

```bash
# Remove from Git (keep local file)
git rm --cached checkpoints/large_checkpoint.pt
git commit -m "chore: remove large checkpoint from git"

# Remove from Git history (rewrite history - use with caution!)
git filter-branch --tree-filter 'rm -f checkpoints/large_checkpoint.pt' HEAD
```

## Directory Structure

```
build_llm_karpathy/
├── checkpoints/
│   ├── .gitkeep                                    # tracked
│   ├── checkpoint_model_step5000_compressed.pt     # tracked (compressed)
│   ├── checkpoint_model_step10000_compressed.pt    # tracked (compressed)
│   ├── checkpoint_model_step500.pt                 # ignored (full size)
│   └── checkpoint_model_step1000.pt                # ignored (full size)
├── CHECKPOINT_GIT_GUIDE.md                         # This file
├── CHECKPOINT_USAGE.md                             # Loading/inference guide
├── compress_checkpoint.py                          # Compression utility
└── .gitignore                                      # Contains checkpoint rules
```

## Troubleshooting

### "File too large" error on push

GitHub has a 100MB file size limit. If you encounter this:

1. Compress the checkpoint:
   ```bash
   python3 scripts/compress_checkpoint.py checkpoint.pt --replace
   ```

2. Or use Git LFS for larger files:
   ```bash
   git lfs track "checkpoints/*.pt"
   git add .gitattributes
   ```

### Accidentally committed large file

First, remove it:
```bash
git rm --cached checkpoints/large_file.pt
echo "checkpoints/*.pt" >> .gitignore
git commit -m "chore: remove and ignore large checkpoint"
git push
```

Then clean up local history (optional):
```bash
rm -rf checkpoints/large_file.pt
```

### Checkpoint won't load after compression

Compressed checkpoints only work for inference. For training resumption, keep the full version:

```bash
# Keep both
python3 scripts/compress_checkpoint.py checkpoint.pt  # Creates _compressed.pt
# Full version: checkpoint.pt (for training)
# Compressed: checkpoint_compressed.pt (for inference)
```
