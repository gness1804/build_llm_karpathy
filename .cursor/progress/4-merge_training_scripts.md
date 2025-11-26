# Plan to Combine `training.py` and `training_resume.py`

This doc outlines how to refactor the two separate training entrypoints into a single, maintainable script (or package entrypoint) that supports both **fresh training** and **resuming from a checkpoint**.

The goal is for a cheaper model (or a future you) to follow these steps and do the heavy refactor, while preserving the current behavior and avoiding subtle bugs.

---

## 1. High-level design

Instead of two large, mostly-duplicated scripts, move to:

- A **single main entrypoint**: e.g. `training.py` that can:
  - start from scratch, or
  - resume from a checkpoint when `CHECKPOINT_PATH` (or similar flag) is set.
- A small **config layer** that:
  - reads environment variables,
  - merges them with checkpoint hyperparameters (if resuming),
  - produces a single `Config` object used by the rest of the code.
- A set of **reusable functions / modules** shared by both flows:
  - device selection
  - data loading + tokenization
  - model construction (from scratch vs GPT-2 + optional LoRA)
  - optimizer / scheduler setup
  - checkpoint save/load
  - training loop
  - generation + output-to-file utilities

Conceptually: `training.py` becomes a thin CLI wrapper around a library-style API, rather than a monolithic script.

---

## 2. Inventory of current responsibilities

### `training.py` (fresh training)

Main responsibilities:

- Read env vars (`MODEL_TYPE`, `USE_LORA`, `TRAINING_DATA_SOURCE`, `TRAINING_STEPS`, LR, batch size, block size, etc.).
- Configure **hyperparameters** (different branches for `TEST_MODE` and for `MODEL_TYPE == 'gpt2'` vs `from_scratch`).
- Select **device** (CUDA / MPS / CPU).
- Load **raw text** from `TRAINING_DATA_SOURCE`.
- For `from_scratch` models: set up tokenization (character / GPT-2 BPE / custom BPE) and build `encode` / `decode`.
- For `gpt2`: instantiate `GPT2Wrapper`, get its tokenizer `encode` / `decode`, encode the dataset.
- Split into `train_data` / `val_data`.
- Define `get_batch` and `estimate_loss`.
- Construct model (Bigram / Bigram+LoRA / GPT-2+LoRA / GPT-2 full) and move to device.
- Initialize optimizer (LoRA-only params vs all params) and optional LR scheduler.
- Manage checkpoint dir / log files.
- Training loop from step `0 .. training_steps-1`.
- Periodic eval + checkpointing.
- Final eval and generation from a prompt.
- Output-to-file via `TeeOutput` and `write_output_file`.

### `training_resume.py` (resume training)

Main responsibilities:

- Read `CHECKPOINT_PATH` and resume-specific env vars (`RESUME_STEPS`, `LEARNING_RATE`, `BATCH_SIZE`, etc.).
- Load checkpoint: `model_state_dict`, `optimizer_state_dict`, `hyperparameters`, `vocab_size`, `block_size`, `batch_size`.
- Derive **model_type**, `use_lora`, embed dims, etc. from checkpoint hyperparameters.
- Optionally override hyperparameters from env.
- Select **device**.
- Load training data, optionally from a new `TRAINING_DATA_SOURCE`.
- Re-create tokenization and `encode`/`decode` based on `model_type` and `tokenization_method`.
- Encode data, split into `train_data` / `val_data`.
- Re-initialize model and load `model_state_dict`.
- Re-create optimizer and load `optimizer_state_dict` (if possible).
- Re-create `TeeOutput`, `get_data_source_name`, `generate_output_filename`, `write_output_file`.
- Define `get_batch`, `estimate_loss`, `save_checkpoint`, `get_model_name`.
- Run a resume-style training loop from `resume_step .. resume_step + RESUME_STEPS - 1`.
- Save checkpoints, logging to a **separate resume log file**.
- Summarize losses, generate final text, and optionally write output file.

There is substantial duplication in:

- device selection
- `TeeOutput` and output file utilities
- `get_data_source_name`, `generate_output_filename`
- tokenization logic for non-GPT-2
- `get_batch`, `estimate_loss`
- model construction for from-scratch models
- generation / decode

---

## 3. Recommended target structure

Refactor into a small internal API under (for example) a `training/` package or just helper modules:

- `config.py`: configuration dataclasses and env → config parsing.
- `data.py`: loading raw text, tokenization, building `encode`/`decode`, building `train_data` / `val_data`.
- `models/__init__.py` (or reuse existing): functions to instantiate the right model given `Config`.
- `train_loop.py`: generic training loop function that can start from any `start_step` and run for `n_steps`.
- `checkpointing.py`: save/load logic and helper functions for building checkpoint file names, logs, etc.
- `io_utils.py`: `TeeOutput`, output file writing, generation helpers.
- `training.py`: main CLI that calls into the above.

You **do not** have to fully modularize on the first pass. A minimal refactor can:

1. Extract shared bits into functions *inside* `training.py`.
2. Slowly move those functions into helpers as a follow-up.

---

## 4. Concrete merge plan (minimal but safe)

### Step 1 – Introduce a `Config` object

Create a small `Config` dataclass in a new file `config.py` or at the top of `training.py`:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # core
    model_type: str
    use_lora: bool
    training_data_source: str
    batch_size: int
    block_size: int
    training_steps: int
    start_step: int  # 0 for fresh, checkpoint step for resume
    learning_rate: float

    # model hyperparams
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    # GPT-2 specifics
    gpt2_model_name: str

    # tokenization
    tokenization_method: str
    custom_vocab_size: Optional[int]

    # checkpointing
    enable_checkpoints: bool
    checkpoint_interval: int
    checkpoint_dir: str
    log_dir: str

    # output
    enable_output_to_file: bool
    output_dir: str

    # misc
    test_mode: bool
```

Then write two loaders:

- `load_config_fresh_from_env()` – reproduces the logic currently in `training.py`.
- `load_config_from_checkpoint(checkpoint, env)` – reproduces the override/merge logic in `training_resume.py`.

In `training.py` you can then do:

- If `CHECKPOINT_PATH` is set → call `load_config_from_checkpoint`.
- Else → call `load_config_fresh_from_env`.

### Step 2 – Unify device selection

Extract device selection into a helper used by both fresh and resume paths:

```python
def select_device(device_env: str = "auto") -> str:
    if device_env != "auto":
        return device_env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

Use this in the unified `training.py` instead of maintaining two versions.

### Step 3 – Unify data loading & tokenization

Create a function that encapsulates everything needed to go from `Config` + `text` → `encode`, `decode`, `train_data`, `val_data`, `vocab_size`:

```python
def prepare_data_and_tokenizer(config: TrainingConfig, raw_text: str, model_type: str, checkpoint_dir: Optional[str] = None):
    """Return (encode, decode, train_data, val_data, vocab_size)."""
    # For model_type == "gpt2", just use GPT2Wrapper.encode/decode and gpt2 vocab size.
    # For from_scratch, reuse the logic from training.py and training_resume.py,
    # including character, GPT-2 BPE, and custom BPE.
```

Then both fresh and resume paths simply:

1. Load raw text from `config.training_data_source`.
2. Call `prepare_data_and_tokenizer`.
3. Store `encode`, `decode`, `train_data`, `val_data`, `vocab_size`.

### Step 4 – Unify model construction

Create a helper:

```python
def build_model(config: TrainingConfig, vocab_size: int, device: str):
    # if config.model_type == "gpt2": use GPT2Wrapper
    # else: use BigramLanguageModel or BigramLanguageModelLoRA
    # move to device and return model
```

- For fresh runs: call `build_model` and skip loading `state_dict`.
- For resume runs: call `build_model` and then `.load_state_dict(model_state_dict)`.

### Step 5 – Unify optimizer & (optional) scheduler

Create a helper to build optimizer given `config` and `model`:

```python
def build_optimizer(config: TrainingConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.use_lora:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()
    return torch.optim.AdamW(params, lr=config.learning_rate)
```

For resume, call this and then, if checkpoint contains `optimizer_state_dict`, load it.

### Step 6 – Generalize `get_batch`, `estimate_loss`, `save_checkpoint`

These functions are nearly identical between scripts. Create single versions:

- `get_batch(split, train_data, val_data, config, device)`
- `estimate_loss(model, train_data, val_data, config, device)`
- `save_checkpoint(step, model, optimizer, config, vocab_size, block_size, batch_size, extra_metadata)`

Use them for both fresh and resume flows.

### Step 7 – Merge training loops into one

Implement one training loop function in the merged `training.py`:

```python
def run_training(config: TrainingConfig,
                model,
                optimizer,
                train_data,
                val_data,
                vocab_size: int,
                start_step: int = 0,
                resume_metadata: Optional[dict] = None):
    # Similar to training.py loop, but:
    # - start at start_step
    # - end at start_step + config.training_steps
    # - use estimate_loss and save_checkpoint helpers
    # - log appropriately (fresh vs resume can be decided by checking start_step)
```

Behavior:

- Fresh run: `start_step = 0`, `training_steps = N`.
- Resume run: `start_step = checkpoint_step`, `training_steps = RESUME_STEPS` (read from env), and `total_steps = start_step + training_steps`.

This removes the need for a second script with its own loop.

### Step 8 – Keep generation & output unified

At the end of `run_training`, or right after it in main:

- Run a single generation block that:
  - For GPT‑2: uses the in-domain prompt (e.g. `"QUESTION: "`).
  - For from-scratch: uses the zero context.
- If `ENABLE_OUTPUT_TO_FILE`, re-use `TeeOutput` and `write_output_file` once.

You already converged on a better generation strategy in `training.py`; use that as the source of truth.

---

## 5. Migration steps for the cheaper model

When you hand this off to a cheaper model, here’s the sequence it should follow inside the repo root:

1. **Back up both scripts**:
   - Copy `training.py` → `training_legacy.py`.
   - Copy `training_resume.py` → `training_resume_legacy.py`.

2. **Introduce `TrainingConfig`** in a new `config.py` or in `training.py`.

3. **Copy shared utilities** from both scripts into `training.py`:
   - `TeeOutput`, `get_data_source_name`, `generate_output_filename`, `write_output_file`, `get_model_name`.

4. **Factor out helpers**:
   - Device selection
   - Data load + tokenization
   - Model construction
   - Optimizer construction
   - `get_batch`, `estimate_loss`, `save_checkpoint`

5. **Implement a unified `run_training`** function that takes `start_step` and `training_steps`.

6. **Implement main logic in `training.py`**:
   - If `CHECKPOINT_PATH` is set → load checkpoint, build config via `load_config_from_checkpoint`, set `start_step` and `training_steps = RESUME_STEPS`, then call `run_training`.
   - Else → build fresh config via `load_config_fresh_from_env`, `start_step = 0`, `training_steps = TRAINING_STEPS`, and call `run_training`.

7. **Port resume-only behaviors** from `training_resume.py` into conditionals in the unified script:
   - Different log file naming for resume vs fresh
   - Extra reporting on start/end loss deltas (optional)

8. **Delete or deprecate `training_resume.py`** once:
   - The merged `training.py` has been tested to:
     - start a fresh run successfully
     - resume from an existing checkpoint successfully
     - produce logs and output comparable to the old scripts.

9. **Update any external workflows** (scripts / workflows that call `training_resume.py`) to use the new unified `training.py` with `CHECKPOINT_PATH` + `RESUME_STEPS`.

---

## 6. Testing checklist after merge

After the cheaper model implements this merge, use these tests before relying on it for long runs:

1. **Fresh GPT‑2 run**
   - No `CHECKPOINT_PATH`.
   - Short run (e.g. 200 steps) with small batch.
   - Confirm loss behavior ~ same as current `training.py`.

2. **Checkpoint resume**
   - Start a fresh run, save a checkpoint at (say) step 200.
   - Re-run `training.py` with `CHECKPOINT_PATH` pointing to that checkpoint and `RESUME_STEPS=300`.
   - Confirm:
     - training starts logging at step 200,
     - final step is 500,
     - losses continue from prior behavior,
     - generated samples look reasonable.

3. **From-scratch bigram model**
   - Run `MODEL_TYPE=from_scratch` both fresh and resumed.

If all three pass, you can safely delete the legacy scripts and continue experimentation (like your GPT‑2 fine-tuning) using the unified entrypoint.
