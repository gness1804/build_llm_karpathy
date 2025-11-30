# Phase 3 Handoff: Unified Training Script

## Status
**Phases 1-2 COMPLETE.** Helper modules extracted and tested.

### What's Been Done
- ✅ `training/config.py` – Config dataclass + loaders (`load_config_fresh_from_env`, `load_config_from_checkpoint`)
- ✅ `training/io_utils.py` – TeeOutput, output file writing, get_model_name, etc.
- ✅ `training/checkpointing.py` – save_checkpoint, load_checkpoint, checkpoint logging
- ✅ `training/data.py` – prepare_data_and_tokenizer, prepare_gpt2_data, get_batch, load_raw_text
- ✅ Backed up originals: `training_legacy.py`, `training_resume_legacy.py`

---

## Phase 3 Task: Create Unified `training.py`

### Overview
Merge `training_legacy.py` (fresh training) and `training_resume_legacy.py` (resume training) into a single `training.py` that:
- Detects fresh vs resume mode based on `CHECKPOINT_PATH` env var
- Uses helper modules to keep code DRY
- Maintains **backward compatibility** with the "Resume training" Warp workflow (which calls `CHECKPOINT_PATH=... python3 training_resume.py`)
- Follows `training_legacy.py` as the source of truth for behavior (it was optimized for coherent text generation)

### Key Design Decisions
1. **Fresh run**: No `CHECKPOINT_PATH` → load config from env, start at step 0
2. **Resume run**: `CHECKPOINT_PATH` set → load checkpoint, build config, start from checkpoint step
3. **Error handling**: If user indicates resume (somehow) but no `CHECKPOINT_PATH`, error out and ask for it
4. **Config merging**: Resume mode merges checkpoint hyperparameters with env variable overrides
5. **Generate prompt**: Use in-domain prompt (`"QUESTION: "`) for GPT-2; use zero token for from-scratch

### Main Structure

```python
# training.py (new unified version)

from training.config import load_config_fresh_from_env, load_config_from_checkpoint
from training.io_utils import *
from training.checkpointing import *
from training.data import *
import torch
import time
import os

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    # Step 1: Detect fresh vs resume mode
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)
    is_resume = checkpoint_path is not None
    
    # Step 2: Load config
    if is_resume:
        if not checkpoint_path:
            print("❌ Error: CHECKPOINT_PATH must be set for resume training")
            sys.exit(1)
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")  # load to CPU first
        config = load_config_from_checkpoint(checkpoint)
    else:
        config = load_config_fresh_from_env()
    
    # Step 3: Select device
    device = _select_device(config.device)  # or use helper from config.py
    
    # Step 4: Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Step 5: Set up stdout capture
    tee_output, original_stdout = setup_stdout_capture(config.enable_output_to_file)
    
    # Step 6: Load data
    raw_text = load_raw_text(config.training_data_source)
    
    # Step 7: Initialize model + tokenizer + data
    # Use different paths for GPT-2 vs from_scratch (see training_legacy.py for reference)
    # For GPT-2: use GPT2Wrapper, get encode/decode from it, prepare_gpt2_data
    # For from_scratch: use prepare_data_and_tokenizer
    
    # Step 8: Build/load model
    # For fresh: call build_model(config)
    # For resume: call build_model(config), then model.load_state_dict(checkpoint["model_state_dict"])
    
    # Step 9: Build/load optimizer
    # For fresh: build_optimizer(config, model)
    # For resume: build_optimizer(config, model), then optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Step 10: Run training loop
    # run_training(
    #     config, model, optimizer, train_data, val_data, vocab_size,
    #     start_step=config.start_step, tee_output=tee_output, device=device
    # )
    
    # Step 11: Generate text and finalize
    # (generation code from training_legacy.py)
    
    # Step 12: Write output file
    # Restore stdout, write output file

if __name__ == "__main__":
    main()
```

### Critical Details from `training_legacy.py` to Preserve

1. **Hyperparameter branching** (lines 230-277):
   - TEST_MODE affects batch_size, block_size, training_steps, eval_interval, eval_iters, n_embd, n_head, n_layer
   - MODEL_TYPE == "gpt2" has different defaults than from_scratch
   - These are already in `config.py`; just use them

2. **GPT-2 model construction** (lines 602-645):
   - Import GPT2Wrapper from `models.gpt2_wrapper`
   - Pass use_lora, lora_rank, lora_alpha, lora_dropout
   - Move to device
   - Get encode/decode/vocab_size from model
   - Encode dataset via `prepare_gpt2_data()`
   - Check block_size vs max_position_embeddings and cap if needed

3. **From-scratch model construction** (lines 647-682):
   - Import BigramLanguageModel or BigramLanguageModelLoRA based on use_lora
   - Call `prepare_data_and_tokenizer()` to get encode/decode/train_data/val_data/vocab_size
   - Move to device

4. **Optimizer** (lines 760-772):
   - If use_lora: optimize only trainable params
   - Otherwise: optimize all model params
   - Use AdamW with learning_rate from config

5. **Training loop** (lines 844-943):
   - Start from `config.start_step` (0 for fresh, checkpoint step for resume)
   - End at `config.start_step + config.training_steps`
   - Call `estimate_loss()` every `config.eval_interval` steps
   - Save checkpoint every `config.checkpoint_interval` steps
   - Print progress every 25 steps (between eval intervals)
   - Track initial_train_loss and last_checkpoint_train_loss for loss deltas

6. **Generation** (lines 981-1013):
   - For GPT-2: use `"QUESTION: "` as prompt (in-domain)
   - For from-scratch: use zero token
   - Extract only new tokens (skip the prompt in output)
   - Temperature: 0.7, top_k: 50, max_new_tokens: 300 (from config)

### Helpers Already Available

#### From `training.config`:
- `TrainingConfig` dataclass
- `load_config_fresh_from_env()`
- `load_config_from_checkpoint(checkpoint)`
- `config_to_dict(config)`

#### From `training.io_utils`:
- `TeeOutput` class
- `setup_stdout_capture(enable)`
- `restore_stdout(original_stdout)`
- `get_data_source_name(path)`
- `get_model_name(model_instance)`
- `generate_output_filename(...)`
- `write_output_file(output_path, hyperparameters, captured_output)`

#### From `training.checkpointing`:
- `load_checkpoint(path, device)`
- `save_checkpoint(step, model, optimizer, config, vocab_size, block_size, batch_size, ...)`
- `create_checkpoint_log_file(log_dir, model_name, source_name, is_resume, resume_step)`
- `write_checkpoint_log(log_file, step, train_loss, val_loss, ...)`
- `close_checkpoint_log(log_file, log_path)`

#### From `training.data`:
- `load_raw_text(data_source)`
- `prepare_data_and_tokenizer(config, raw_text, model_type)` – for from_scratch
- `prepare_gpt2_data(encode_fn, raw_text)` – for GPT-2
- `get_batch(split, train_data, val_data, batch_size, block_size, device)`

### Validation Commands (Run AFTER Phase 3)

**Fresh GPT-2 (200 steps):**
```bash
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
MODEL_TYPE=gpt2 \
USE_LORA=False \
TRAINING_STEPS=200 \
LEARNING_RATE=5e-6 \
BLOCK_SIZE=128 \
BATCH_SIZE=8 \
ENABLE_CHECKPOINTS=False \
USE_LR_WARMUP=False \
ENABLE_OUTPUT_TO_FILE=False \
python3 training.py
```

**Resume from checkpoint (100 additional steps):**
```bash
CHECKPOINT_PATH=checkpoints/checkpoint_gpt2_training_data_final_merged_step000100_XXXXXXX.pt \
RESUME_STEPS=100 \
python3 training.py
```

**From-scratch model (100 steps, character tokenization):**
```bash
TRAINING_DATA_SOURCE=sources/training_data_final_merged.md \
MODEL_TYPE=from_scratch \
TRAINING_STEPS=100 \
BATCH_SIZE=16 \
ENABLE_CHECKPOINTS=False \
ENABLE_OUTPUT_TO_FILE=False \
python3 training.py
```

### Backward Compatibility
The new script maintains compatibility with the existing Warp workflow:
```
CHECKPOINT_PATH={{checkpoint_path}} python3 training.py
```
(This was previously `training_resume.py`, now just `training.py`)

### Notes
- `training_legacy.py` should be the source of truth for behavior; copy its structure/logic
- Keep all the print statements and emojis for user feedback
- Preserve gradient clipping logic (1.0 for GPT-2 full FT, 2.0 for LoRA)
- Keep eval_interval logic: eval at step % eval_interval == 0
- Make sure loss tracking for "Net loss change since beginning/checkpoint" is preserved

---

## Next Steps for New Agent
1. Read this document carefully
2. Skim `training_legacy.py` to understand overall structure (especially the three main phases: setup, loop, generation)
3. Create `training.py` by:
   - Copy the overall structure from `training_legacy.py`
   - Replace duplicated utility code with imports from helpers
   - Add fresh/resume detection at the top
   - Test with the validation commands above
4. Once working, delete the legacy files (but keep backups for reference)

Good luck!

<!-- DONE -->
