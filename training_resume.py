# ruff: noqa: E731
"""
Resume training from a checkpoint with optional new data and hyperparameters.

Usage:
    CHECKPOINT_PATH=checkpoints/checkpoint.pt python3 training_resume.py

Optional environment variables:
    TRAINING_DATA_SOURCE - Path to new training data (default: from checkpoint)
    RESUME_STEPS - Number of additional steps to train (default: 5000)
    LEARNING_RATE - Learning rate override (default: from checkpoint or 3e-4)
    BATCH_SIZE - Batch size override (default: from checkpoint)
    CHECKPOINT_INTERVAL - How often to save checkpoints (default: 500)
    ENABLE_CHECKPOINTS - Enable checkpoint saving (default: true)
"""

import torch
import os
import sys
import time
from datetime import datetime
from io import StringIO

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", None)
DEFAULT_TRAINING_DATA_SOURCE = "sources/carolyn_hax/carolyn_hax_merged_cleaned.md"
TRAINING_DATA_SOURCE = os.environ.get(
    "TRAINING_DATA_SOURCE", DEFAULT_TRAINING_DATA_SOURCE
)
RESUME_STEPS = int(os.environ.get("RESUME_STEPS", "5000"))
print(f"RESUME_STEPS: {RESUME_STEPS}")
LEARNING_RATE_OVERRIDE = os.environ.get("LEARNING_RATE", None)
BATCH_SIZE_OVERRIDE = os.environ.get("BATCH_SIZE", None)

# Checkpoint configuration
ENABLE_CHECKPOINTS = os.environ.get("ENABLE_CHECKPOINTS", "true").lower() == "true"
CHECKPOINT_INTERVAL = int(os.environ.get("CHECKPOINT_INTERVAL", "500"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
ENABLE_OUTPUT_TO_FILE = (
    os.environ.get("ENABLE_OUTPUT_TO_FILE", "true").lower() == "true"
)

DEVICE = os.environ.get("DEVICE", "auto")

# ============================================================================
# DEVICE SETUP
# ============================================================================

if DEVICE == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("⚠️  Using CPU (slow)")
else:
    device = DEVICE
    print(f"Using device: {device}")

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

if not CHECKPOINT_PATH:
    print("❌ Error: CHECKPOINT_PATH environment variable not set")
    print("Usage: CHECKPOINT_PATH=/path/to/checkpoint.pt python3 training_resume.py")
    sys.exit(1)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)

# Extract checkpoint data
resume_step = checkpoint.get("step", 0)
model_state_dict = checkpoint.get("model_state_dict")
optimizer_state_dict = checkpoint.get("optimizer_state_dict")
hyperparameters = checkpoint.get("hyperparameters", {})
vocab_size = checkpoint.get("vocab_size")
block_size = checkpoint.get("block_size")
batch_size_checkpoint = checkpoint.get("batch_size")

if not model_state_dict:
    print("❌ Error: Checkpoint does not contain model_state_dict")
    sys.exit(1)

print(f"✅ Checkpoint loaded (trained for {resume_step} steps)")

# ============================================================================
# HYPERPARAMETER SETUP
# ============================================================================

model_type = hyperparameters.get("model_type", "from_scratch").lower()
use_lora = hyperparameters.get("use_lora", False)

# Use overrides or checkpoint values
batch_size = int(BATCH_SIZE_OVERRIDE) if BATCH_SIZE_OVERRIDE else batch_size_checkpoint
learning_rate = (
    float(LEARNING_RATE_OVERRIDE)
    if LEARNING_RATE_OVERRIDE
    else hyperparameters.get("learning_rate", 3e-4)
)
n_embd = hyperparameters.get("n_embd", 384)
n_head = hyperparameters.get("n_head", 6)
n_layer = hyperparameters.get("n_layer", 6)
dropout = hyperparameters.get("dropout", 0.2)

# Use new data source or fall back to checkpoint data source
if TRAINING_DATA_SOURCE:
    data_source = TRAINING_DATA_SOURCE
    print(f"\n📝 Using new training data: {data_source}")
else:
    # Try to infer from checkpoint metadata or use default
    data_source = hyperparameters.get(
        "training_data_source", DEFAULT_TRAINING_DATA_SOURCE
    )
    print(f"\n📝 Using original training data: {data_source}")

print("\n📊 Training Configuration:")
print(f"   Model type: {model_type}")
print(f"   LoRA enabled: {use_lora}")
print(f"   Resuming from step: {resume_step}")
print(f"   Additional steps: {RESUME_STEPS}")
print(f"   Total steps will be: {resume_step + RESUME_STEPS}")
print(f"   Learning rate: {learning_rate}")
print(f"   Batch size: {batch_size}")
print(f"   Block size: {block_size}")
print(f"   Training data source: {data_source}")

# ============================================================================
# DATA LOADING
# ============================================================================

# (Data loading logic removed from here as it was moved up)

if not os.path.exists(data_source):
    print(f"❌ Error: Training data not found at {data_source}")
    sys.exit(1)

# Load training data
with open(data_source, "r", encoding="utf-8") as f:
    text = f.read()

print(f"   Dataset size: {len(text):,} characters")

# ============================================================================
# MODEL INITIALIZATION (needed first for GPT-2 to get encode/decode)
# ============================================================================

print("\n🤖 Initializing model...")

# Initialize encode/decode functions
encode = None
decode = None

if model_type == "gpt2":
    from models.gpt2_wrapper import GPT2Wrapper

    gpt2_model_name = hyperparameters.get("gpt2_model_name", "gpt2")
    lora_rank = hyperparameters.get("lora_rank", 8)
    lora_alpha = hyperparameters.get("lora_alpha", 16.0)
    lora_dropout = hyperparameters.get("lora_dropout", 0.0)

    model = GPT2Wrapper(
        model_name=gpt2_model_name,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
    )

    encode = model.encode
    decode = model.decode
    tokenization_method = "gpt2"
    print(f"\n🔤 Tokenization: {tokenization_method}")

# ============================================================================
# TOKENIZATION SETUP (for non-GPT-2 models)
# ============================================================================

if model_type != "gpt2":
    tokenization_method = hyperparameters.get("tokenization_method", "character")
    print(f"\n🔤 Tokenization: {tokenization_method}")

    if tokenization_method == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s)
        decode = lambda ids: enc.decode(ids)

    elif tokenization_method == "custom_bpe":
        try:
            from tokenizers import Tokenizer

            checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
            tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

            if os.path.exists(tokenizer_path):
                tokenizer = Tokenizer.from_file(tokenizer_path)
                encode = lambda s: tokenizer.encode(s).ids
                decode = lambda ids: tokenizer.decode(ids)
                print("   ✅ Loaded custom BPE tokenizer")
            else:
                print("   ⚠️  Tokenizer not found, falling back to character-level")
                tokenization_method = "character"
        except ImportError:
            print("   ⚠️  tokenizers not installed, using character-level")
            tokenization_method = "character"

    if tokenization_method == "character":
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda ids: "".join([itos.get(i, "?") for i in ids])

# Encode data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"   Train samples: {len(train_data):,}, Val samples: {len(val_data):,}")

# ============================================================================
# MODEL INITIALIZATION (for from_scratch models)
# ============================================================================

if model_type == "from_scratch":
    from models.bigram_lm_v2 import BigramLanguageModel
    from models.bigram_lm_v2_lora import BigramLanguageModelLoRA

    lora_rank = hyperparameters.get("lora_rank", 8)
    lora_alpha = hyperparameters.get("lora_alpha", 16.0)
    lora_dropout = hyperparameters.get("lora_dropout", 0.0)

    if use_lora:
        model = BigramLanguageModelLoRA(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            device=device,
            dropout=dropout,
            n_head=n_head,
            n_layer=n_layer,
            use_lora=True,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    else:
        model = BigramLanguageModel(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            device=device,
            dropout=dropout,
            n_head=n_head,
            n_layer=n_layer,
        )
elif model_type != "gpt2":
    print(f"❌ Error: Unknown model type: {model_type}")
    sys.exit(1)

# Load model state
model.load_state_dict(model_state_dict)
model.to(device)

print(f"✅ Model loaded and moved to {device}")

# ============================================================================
# OPTIMIZER SETUP
# ============================================================================

if use_lora:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Load optimizer state if available
if optimizer_state_dict:
    try:
        optimizer.load_state_dict(optimizer_state_dict)
        print("✅ Optimizer state loaded")
    except Exception as e:
        print(f"⚠️  Could not load optimizer state: {e}")
        print("   Starting with fresh optimizer")

# ============================================================================
# OUTPUT TO FILE UTILITIES
# ============================================================================


class TeeOutput:
    """Capture stdout while still displaying to terminal"""

    def __init__(self, *files):
        self.files = files
        self.buffer = StringIO()

    def write(self, obj):
        # Write to all files (stdout + buffer)
        for f in self.files:
            f.write(obj)
            f.flush()
        # Also write to buffer for later retrieval
        self.buffer.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def get_data_source_name(training_data_source):
    """Extract data source name without extension from path"""
    # Handle both relative and absolute paths
    source_path = os.path.normpath(training_data_source)
    # Get filename with extension
    filename = os.path.basename(source_path)
    # Remove extension
    source_name = os.path.splitext(filename)[0]
    return source_name


def generate_output_filename(
    model_name,
    source_name,
    vocab_size,
    training_steps,
    test_mode,
    use_lora=False,
    lora_rank=None,
    lora_alpha=None,
    model_type="from_scratch",
    gpt2_model_name=None,
):
    """Generate output filename with structured naming convention"""
    # Format timestamp as MMDDYYYY_HHMMSS
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

    # Construct filename components
    components = [
        "build_llm_output",
        model_name,
        source_name,
        str(vocab_size),
        str(training_steps),
        f"test={str(test_mode).lower()}",
    ]

    # Add model type information
    if model_type == "gpt2":
        if gpt2_model_name:
            components.append(f"gpt2_{gpt2_model_name}")
        else:
            components.append("gpt2")
    else:
        components.append("from_scratch")

    # Add LoRA information if used
    if use_lora:
        lora_info = f"lora_r{lora_rank}_a{lora_alpha}"
        components.append(lora_info)
    else:
        components.append("full_ft")  # full fine-tuning

    components.extend(["OUTPUT", timestamp])

    # Join with underscores (snake_case)
    filename = "_".join(components) + ".txt"
    return filename


def write_output_file(output_path, hyperparameters, captured_output):
    """Write hyperparameters and captured output to file"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write hyperparameters section
            f.write("HYPERPARAMETERS\n")
            f.write("=" * 16 + "\n")
            for key, value in hyperparameters.items():
                f.write(f"{key} = {value}\n")

            f.write("\n")

            # Write output section
            f.write("OUTPUT\n")
            f.write("=" * 6 + "\n")
            f.write(captured_output)

        return True
    except Exception as e:
        print(f"⚠️  Error writing output file: {e}")
        return False


# Create checkpoint and output directories
if ENABLE_CHECKPOINTS:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if ENABLE_OUTPUT_TO_FILE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize stdout capture if output to file is enabled
tee_output = None
original_stdout = None
if ENABLE_OUTPUT_TO_FILE:
    original_stdout = sys.stdout
    tee_output = TeeOutput(sys.stdout)
    sys.stdout = tee_output

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_iters = hyperparameters.get("eval_iters", 200)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(step, model, optimizer):
    """Save resumed training checkpoint"""
    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparameters": hyperparameters,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "batch_size": batch_size,
    }

    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    checkpoint_name = f"checkpoint_resumed_step{step:06d}_{timestamp}.pt"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    try:
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    except Exception as e:
        print(f"⚠️  Error saving checkpoint: {e}")
        return None


def get_model_name(model_instance):
    """Extract model name from model class"""
    if model_type == "gpt2":
        if hasattr(model_instance, "model_name"):
            return model_instance.model_name.replace("-", "")
        return "gpt2"

    class_name = model_instance.__class__.__name__
    if "LanguageModel" in class_name:
        return class_name.replace("LanguageModel", "").lower()
    if "Wrapper" in class_name:
        return class_name.replace("Wrapper", "").lower()
    return class_name.lower()


# ============================================================================
# TRAINING LOOP
# ============================================================================

eval_interval = hyperparameters.get("eval_interval", 500)

print(f"\n{'=' * 60}")
print("🚀 RESUMING TRAINING")
print(f"{'=' * 60}")
print(f"Starting from step {resume_step}")
print(f"Training for {RESUME_STEPS} additional steps")
print(f"Total training steps: {resume_step + RESUME_STEPS}")
print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} steps")
print(f"{'=' * 60}\n")

start_time = time.time()
total_steps = resume_step + RESUME_STEPS

for step in range(resume_step, total_steps):
    # Evaluate periodically
    if step % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        steps_per_sec = (step - resume_step) / elapsed if elapsed > 0 else 0
        progress_pct = ((step - resume_step) / RESUME_STEPS) * 100
        print(
            f"step {step}/{total_steps} ({progress_pct:.1f}%): train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | {elapsed:.1f}s ({steps_per_sec:.2f} steps/sec)"
        )

        # Save checkpoint
        if (
            ENABLE_CHECKPOINTS
            and step % CHECKPOINT_INTERVAL == 0
            and step > resume_step
        ):
            checkpoint_path = save_checkpoint(step, model, optimizer)
            if checkpoint_path:
                print(f"   💾 Checkpoint saved: {checkpoint_path}")

    # Progress indicator
    elif step > resume_step and (step - resume_step) % 25 == 0:
        elapsed = time.time() - start_time
        steps_per_sec = (step - resume_step) / elapsed if elapsed > 0 else 0
        progress_pct = ((step - resume_step) / RESUME_STEPS) * 100
        eta_seconds = ((total_steps - step) / steps_per_sec) if steps_per_sec > 0 else 0
        eta_minutes = eta_seconds / 60
        print(
            f"step {step}/{total_steps} ({progress_pct:.1f}%) | {steps_per_sec:.2f} steps/sec | ETA: {eta_minutes:.1f}m"
        )

    # Training step
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\n{'=' * 60}")
print("✅ Training complete!")
print(f"{'=' * 60}")
total_time = time.time() - start_time
steps_completed = total_steps - resume_step
print(f"Steps completed: {steps_completed}")
print(
    f"Total training time: {total_time:.1f}s ({steps_completed/total_time:.2f} steps/sec)"
)
print(f"Final loss: {loss.item():.4f}")

# Save final checkpoint
if ENABLE_CHECKPOINTS:
    final_checkpoint_path = save_checkpoint(total_steps, model, optimizer)
    if final_checkpoint_path:
        print(f"✅ Final checkpoint saved: {final_checkpoint_path}")

# Summarize each loss that appeared and the differences between them
losses = estimate_loss()
print(f"Train loss at beginning: {losses['train']:.4f}")
print(f"Val loss at beginning: {losses['val']:.4f}")
print(f"Train loss at end: {loss.item():.4f}")
print(f"Val loss at end: {losses['val']:.4f}")
print(
    f"Difference between val loss at beginning and val loss at end: {losses['val'] - loss.item():.4f}"
)  # This is the improvement in val loss
print(
    f"Difference between val loss at end and val loss at beginning as a percentage of the beginning loss: {(losses['val'] - loss.item()) / losses['val'] * 100:.2f}%"
)

print(f"{'=' * 60}\n")

# Generate sample text
print("Generating sample text...")
print("=" * 50)

if model_type == "gpt2":
    context = torch.tensor([[model.tokenizer.bos_token_id or 0]], dtype=torch.long).to(
        device
    )
else:
    context = torch.zeros((1, 1), dtype=torch.long).to(device)

max_new_tokens = hyperparameters.get("max_new_tokens", 300)

with torch.no_grad():
    generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)

if model_type == "gpt2":
    if generated_tokens.shape[1] > context.shape[1]:
        new_tokens = generated_tokens[0, context.shape[1] :].tolist()
    else:
        new_tokens = generated_tokens[0].tolist()
    generated_text = decode(new_tokens)
else:
    generated_text = decode(generated_tokens[0].tolist())

print(generated_text)
print("=" * 50)

# ============================================================================
# WRITE OUTPUT TO FILE
# ============================================================================

if ENABLE_OUTPUT_TO_FILE:
    # Restore stdout
    sys.stdout = original_stdout

    # Get captured output
    captured_output = tee_output.getvalue()

    # Extract information for filename generation
    model_name = get_model_name(model)
    source_name = get_data_source_name(data_source)
    test_mode = hyperparameters.get("test_mode", False)
    gpt2_model_name = (
        hyperparameters.get("gpt2_model_name") if model_type == "gpt2" else None
    )
    lora_rank = hyperparameters.get("lora_rank") if use_lora else None
    lora_alpha = hyperparameters.get("lora_alpha") if use_lora else None

    # Generate filename
    filename = generate_output_filename(
        model_name=model_name,
        source_name=source_name,
        vocab_size=vocab_size,
        training_steps=total_steps,  # Total steps including resumed steps
        test_mode=test_mode,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        model_type=model_type,
        gpt2_model_name=gpt2_model_name,
    )

    # Construct full output path
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Update hyperparameters with resume information
    resume_hyperparameters = hyperparameters.copy()
    resume_hyperparameters["resume_step"] = resume_step
    resume_hyperparameters["resume_steps"] = RESUME_STEPS
    resume_hyperparameters["total_steps"] = total_steps
    resume_hyperparameters["learning_rate"] = learning_rate
    resume_hyperparameters["batch_size"] = batch_size

    # Write output file
    if write_output_file(output_path, resume_hyperparameters, captured_output):
        print(f"\n✅ Output written to: {output_path}")
    else:
        print("\n⚠️  Failed to write output file")

print("\n✅ Resume training complete!")
