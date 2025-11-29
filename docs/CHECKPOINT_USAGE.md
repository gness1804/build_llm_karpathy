# Checkpoint Usage Guide

This guide explains how to save and load model checkpoints for inference and resuming training.

## Saving Checkpoints

### During Training

To save checkpoints periodically during training, use the `ENABLE_CHECKPOINTS` environment variable:

```bash
ENABLE_CHECKPOINTS=true \
CHECKPOINT_INTERVAL=500 \
CHECKPOINT_DIR=checkpoints \
MODEL_TYPE=gpt2 \
python3 training.py
```

**Checkpoint variables:**
- `ENABLE_CHECKPOINTS=true` - Enable checkpoint saving (default: false)
- `CHECKPOINT_INTERVAL=500` - Save every N steps (default: 500)
- `CHECKPOINT_DIR=checkpoints` - Directory to save checkpoints (default: `checkpoints/`)

Checkpoints are saved with the format:
```
checkpoint_{model}_{source}_step{N}_{timestamp}.pt
```

Example:
```
checkpoint_bigram_carolyn_hax_step0500_11102025_231500.pt
```

### Checkpoint Contents

Each checkpoint contains:
- **model_state_dict**: Trained model weights
- **optimizer_state_dict**: Optimizer state (for resuming training)
- **hyperparameters**: All training configuration
- **vocab_size**: Vocabulary size
- **block_size**: Context window size
- **batch_size**: Training batch size
- **step**: Training step number

## Loading and Using Checkpoints

Use `scripts/load_checkpoint.py` for both inference and training resumption:

### Inference Mode

Generate text using a trained model:

```bash
CHECKPOINT_PATH=checkpoints/checkpoint_bigram_carolyn_hax_step0500_11102025_231500.pt \
MODE=inference \
MAX_NEW_TOKENS=500 \
python3 scripts/load_checkpoint.py
```

**Optional parameters:**
- `PROMPT="Your prompt here"` - Start generation with a specific prompt (default: empty/random)
- `MAX_NEW_TOKENS=300` - Number of tokens to generate (default: 300)
- `DEVICE=auto` - Device to use: `auto`, `cpu`, `cuda`, `mps` (default: auto-detect)

**Examples:**

Generate without prompt:
```bash
CHECKPOINT_PATH=checkpoints/checkpoint_bigram_carolyn_hax_step5000_11102025_231500.pt \
MODE=inference \
python3 scripts/load_checkpoint.py
```

Generate with prompt:
```bash
CHECKPOINT_PATH=checkpoints/checkpoint_bigram_carolyn_hax_step5000_11102025_231500.pt \
MODE=inference \
PROMPT="Dear Carolyn," \
MAX_NEW_TOKENS=500 \
python3 scripts/load_checkpoint.py
```

Generate on CPU (useful if GPU memory is limited):
```bash
CHECKPOINT_PATH=checkpoints/checkpoint_bigram_carolyn_hax_step5000_11102025_231500.pt \
MODE=inference \
DEVICE=cpu \
python3 scripts/load_checkpoint.py
```

### Resume Training Mode

Resume training from a checkpoint with new data or hyperparameters:

```bash
CHECKPOINT_PATH=checkpoints/checkpoint_bigram_carolyn_hax_step5000_11102025_231500.pt \
MODE=resume \
RESUME_TRAINING_STEPS=2000 \
RESUME_LEARNING_RATE=1e-4 \
python3 scripts/load_checkpoint.py
```

**Parameters:**
- `RESUME_TRAINING_STEPS=5000` - Number of additional steps to train (default: 5000)
- `RESUME_LEARNING_RATE=3e-4` - Learning rate for resumed training (default: 3e-4)

This will display information about the checkpoint and prepare it for resuming training.

## Important Notes

### Tokenization Support

**GPT-2 models**: Fully supported - uses built-in tokenizer

**Custom BPE tokenizers**: 
- The script looks for `tokenizer.json` in the same directory as the checkpoint
- You may need to save the tokenizer separately if using custom BPE
- Falls back to character-level if not found

**Character-level tokenization**:
- ⚠️ Limited support - requires vocab to be reconstructed
- May not perfectly match original if special characters are present
- Recommended: Use GPT-2 or custom BPE tokenization for better checkpoint compatibility

### Checkpoint File Size

- **from_scratch models**: 1-10 MB depending on model size
- **GPT-2 base**: ~500 MB
- **GPT-2 medium**: ~1.5 GB
- **GPT-2 large**: ~3 GB
- **GPT-2 XL**: ~6 GB

LoRA checkpoints are much smaller (same size as non-LoRA for file format, but only a small subset of weights)

## Finding Your Checkpoints

List all saved checkpoints:

```bash
ls -lh checkpoints/
```

Find checkpoints from a specific model/data source:

```bash
ls checkpoints/ | grep "bigram_carolyn"
```

Find checkpoints from a specific step:

```bash
ls checkpoints/ | grep "step5000"
```

## Troubleshooting

**"Checkpoint not found"**
- Verify the full path to the checkpoint file
- Check that `ENABLE_CHECKPOINTS=true` was used during training
- List checkpoints with: `ls checkpoints/`

**"Checkpoint does not contain model_state_dict"**
- The file may be corrupted
- Try using a different checkpoint from an earlier step
- Re-run training with checkpoints enabled

**Device errors**
- Use `DEVICE=cpu` as a fallback
- Check CUDA/MPS availability on your system

**Memory errors during inference**
- Use `DEVICE=cpu` instead of GPU
- Reduce `MAX_NEW_TOKENS` value
- Use a smaller checkpoint (from an earlier training step)

## Workflow Examples

### Train, Save, and Generate

```bash
# Step 1: Train with checkpoints
ENABLE_CHECKPOINTS=true \
CHECKPOINT_INTERVAL=500 \
MODEL_TYPE=gpt2 \
TRAINING_STEPS=2000 \
python3 training.py

# Step 2: List available checkpoints
ls checkpoints/ | tail -5

# Step 3: Generate text with the final checkpoint
CHECKPOINT_PATH=checkpoints/checkpoint_gpt2_carolyn_hax_step2000_*.pt \
MODE=inference \
PROMPT="Dear Carolyn, I have a problem..." \
python3 load_checkpoint.py
```

### Continue Training with More Data

```bash
# Step 1: Train initial model with checkpoints
ENABLE_CHECKPOINTS=true MODEL_TYPE=gpt2 TRAINING_STEPS=1000 python3 training.py

# Step 2: Find the final checkpoint
CHECKPOINT=$(ls checkpoints/ | sort | tail -1)

# Step 3: Inspect checkpoint before resuming
CHECKPOINT_PATH=checkpoints/$CHECKPOINT MODE=resume python3 scripts/load_checkpoint.py

# Step 4: Resume training (implementation coming)
# CHECKPOINT_PATH=checkpoints/$CHECKPOINT TRAINING_DATA_SOURCE=new_data.md python3 training_resume.py
```

### A/B Testing Different Models

```bash
# Train GPT-2 base
ENABLE_CHECKPOINTS=true MODEL_TYPE=gpt2 GPT2_MODEL_NAME=gpt2 TRAINING_STEPS=1000 python3 training.py

# Train GPT-2 medium
ENABLE_CHECKPOINTS=true MODEL_TYPE=gpt2 GPT2_MODEL_NAME=gpt2-medium TRAINING_STEPS=1000 python3 training.py

# Generate with both for comparison
CHECKPOINT_PATH=checkpoints/checkpoint_gpt2_*_gpt2_step1000*.pt MODE=inference python3 scripts/load_checkpoint.py

CHECKPOINT_PATH=checkpoints/checkpoint_gpt2_*_gpt2medium_step1000*.pt MODE=inference python3 scripts/load_checkpoint.py
```
