"""
Bigram Language Model - Building an LLM from Scratch
Following Andrej Karpathy's tutorial
"""

import tiktoken
from models.bigram_lm_v2 import BigramLanguageModel

import time
import torch
import os
import sys
from datetime import datetime
from io import StringIO


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string (hours, minutes, seconds).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "1h 23m 45s" or "23m 45s" or "45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:  # Always show seconds if no hours/minutes
        parts.append(f"{secs}s")

    return " ".join(parts)


is_test_mode = os.environ.get("TEST_MODE", "False")

# LoRA configuration
USE_LORA = os.environ.get("USE_LORA", "False").lower() == "true"
LORA_RANK = int(os.environ.get("LORA_RANK", "8"))
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "16.0"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))

# Model selection: "from_scratch" or "gpt2"
MODEL_TYPE = os.environ.get("MODEL_TYPE", "from_scratch").lower()
GPT2_MODEL_NAME = os.environ.get(
    "GPT2_MODEL_NAME", "gpt2"
)  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

# Checkpoint configuration
ENABLE_CHECKPOINTS = os.environ.get("ENABLE_CHECKPOINTS", "False").lower() == "true"
CHECKPOINT_INTERVAL = int(
    os.environ.get("CHECKPOINT_INTERVAL", "500")
)  # Save every N steps
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")


# Custom BPE tokenization uses HuggingFace tokenizers library
# No need to import here - imported when custom_bpe method is selected

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Set to True for fast testing with smaller model, False for full training
TEST_MODE = is_test_mode == "True"
TRAINING_DATA_SOURCE = os.environ.get(
    "TRAINING_DATA_SOURCE", "sources/carolyn_hax_103125_chat.md"
)
TOKENIZATION_METHOD = os.environ.get(
    "TOKENIZATION_METHOD", "character"
)  # "character", "gpt2", or "custom_bpe"
CUSTOM_VOCAB_SIZE = os.environ.get("CUSTOM_VOCAB_SIZE", None)

# Output to file configuration
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
ENABLE_OUTPUT_TO_FILE = (
    os.environ.get("ENABLE_OUTPUT_TO_FILE", "True").lower() == "true"
)

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


# Create output directory if it doesn't exist
if ENABLE_OUTPUT_TO_FILE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create checkpoint directory if it doesn't exist
if ENABLE_CHECKPOINTS:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize stdout capture if output to file is enabled
tee_output = None
original_stdout = None
if ENABLE_OUTPUT_TO_FILE:
    original_stdout = sys.stdout
    tee_output = TeeOutput(sys.stdout)
    sys.stdout = tee_output

# Set random seed for reproducibility
torch.manual_seed(1337)

# Model hyperparameters
# Allow override of training_steps via environment variable for quick testing
TRAINING_STEPS_OVERRIDE = os.environ.get("TRAINING_STEPS", None)

if TEST_MODE:
    # Fast configuration for testing and debugging
    batch_size = 32  # Reduced from 64
    block_size = 64  # Reduced from 256 (16x less attention computation!)
    training_steps = (
        int(TRAINING_STEPS_OVERRIDE) if TRAINING_STEPS_OVERRIDE else 1000
    )  # Reduced from 5000
    eval_interval = 100  # Evaluate more frequently
    learning_rate = 3e-4  # Learning rate for optimizer
    eval_iters = 50  # Reduced from 200
    n_embd = 128  # Reduced from 384
    n_head = 4  # Reduced from 6
    n_layer = 3  # Reduced from 6 (half the layers!)
    dropout = 0.2  # Dropout rate for self-attention
    print("🔬 TEST MODE: Using reduced hyperparameters for fast training")
else:
    # Full configuration for production training (aggressively optimized for Apple Silicon)
    batch_size = 64  # Reduced from 64 for better M4 performance
    block_size = 128  # Further reduced from 128 (4x less attention computation)
    training_steps = (
        int(TRAINING_STEPS_OVERRIDE) if TRAINING_STEPS_OVERRIDE else 5000
    )  # Number of training iterations
    eval_interval = 500  # More frequent feedback (reduced from 500)
    learning_rate = 3e-4  # Learning rate for optimizer
    eval_iters = 50  # Further reduced from 50 for faster eval
    n_embd = 256  # Further reduced from 256 for Apple Silicon
    n_head = 4  # Further reduced from 4 for Apple Silicon
    n_layer = 4  # Further reduced from 4 for Apple Silicon
    dropout = 0.2  # Dropout rate for self-attention
    print(
        "🚀 FULL MODE: Using production hyperparameters (aggressively optimized for M4)"
    )

# Device selection: prioritize MPS (Apple Silicon GPU) > CUDA > CPU
if torch.cuda.is_available():
    device = "cuda"
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
else:
    device = "cpu"
    print("⚠️  Using CPU (slow) - consider using MPS if available")

# Generation settings
max_new_tokens = 300  # Number of characters to generate

# Collect hyperparameters for output file
hyperparameters = {
    "batch_size": batch_size,
    "block_size": block_size,
    "training_steps": training_steps,
    "eval_interval": eval_interval,
    "learning_rate": learning_rate,
    "eval_iters": eval_iters,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "max_new_tokens": max_new_tokens,
    "device": device,
    "tokenization_method": TOKENIZATION_METHOD,
    "test_mode": TEST_MODE,
    "use_lora": USE_LORA,
    "model_type": MODEL_TYPE,
    "training_data_source": TRAINING_DATA_SOURCE,
}

if MODEL_TYPE == "gpt2":
    hyperparameters["gpt2_model_name"] = GPT2_MODEL_NAME

if USE_LORA:
    hyperparameters.update(
        {
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
        }
    )

print(f"Device: {device}")
print(f"Model size: {n_layer} layers, {n_embd} embedding dims, {n_head} heads")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Load training data
with open(TRAINING_DATA_SOURCE, "r", encoding="utf-8") as f:
    text = f.read()

# TOKENIZATION OPTIONS (for from_scratch models):
# Option 1: Character-level (original - fastest, simplest, best for small datasets)
# Option 2: GPT-2 BPE via tiktoken (industry standard but large vocab: 50,257 tokens)
# Option 3: Custom BPE tokenizer (balanced - train on your dataset for optimal vocab size)

# If using GPT-2 model, skip custom tokenization (GPT-2 has its own tokenizer)
if MODEL_TYPE != "gpt2":
    # Choose tokenization method:
    if TOKENIZATION_METHOD == "gpt2":
        # GPT-2 BPE: Industry standard but large vocabulary (50,257 tokens)
        # Better for large datasets, but slower and higher loss for small datasets
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab

        def encode(s):
            return enc.encode(s)

        def decode(token_ids):
            return enc.decode(token_ids)

        print(f"📝 Using GPT-2 BPE tokenization (vocab_size={vocab_size:,})")
        print("⚠️  Note: Large vocab may cause higher loss and slower training")

    elif TOKENIZATION_METHOD == "custom_bpe":
        # Custom BPE: Train a tokenizer on your dataset (recommended for balanced approach)
        # Uses HuggingFace tokenizers library: pip install tokenizers
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
            import math

            # Automatically scale vocab size based on dataset characteristics
            # This balances efficiency (compression) with model size (parameters)
            # Can be overridden with CUSTOM_VOCAB_SIZE environment variable
            manual_vocab_size = CUSTOM_VOCAB_SIZE

            dataset_size_bytes = len(text.encode("utf-8"))
            dataset_size_mb = dataset_size_bytes / (1024 * 1024)
            num_unique_chars = len(set(text))

            if manual_vocab_size and manual_vocab_size.lower() != "none":
                target_vocab_size = int(CUSTOM_VOCAB_SIZE)
                print(f"🔧 Using manual vocab_size override: {target_vocab_size:,}")
            else:
                # Calculate target vocab size using heuristics:
                # 1. Base size on dataset size (larger datasets benefit from larger vocabs)
                # 2. Minimum vocab size ensures good coverage
                # 3. Maximum vocab size caps model complexity

                # Heuristic: vocab_size scales logarithmically with dataset size
                # Formula: base_vocab + log2(dataset_size_mb) * scale_factor
                # This ensures:
                # - Small datasets (< 1MB): ~500-1000 tokens
                # - Medium datasets (1-10MB): ~1000-3000 tokens
                # - Large datasets (> 10MB): ~3000-8000 tokens

                base_vocab = 500
                scale_factor = 200

                # Calculate target vocab size
                if dataset_size_mb < 0.1:
                    # Very small datasets: use smaller vocab
                    target_vocab_size = max(256, min(1000, num_unique_chars * 4))
                elif dataset_size_mb < 1.0:
                    # Small datasets: 500-1500 tokens
                    target_vocab_size = base_vocab + int(
                        math.log2(max(0.1, dataset_size_mb)) * scale_factor
                    )
                elif dataset_size_mb < 10.0:
                    # Medium datasets: 1000-3000 tokens
                    target_vocab_size = base_vocab + int(
                        math.log2(max(1.0, dataset_size_mb)) * scale_factor
                    )
                else:
                    # Large datasets: 3000-8000 tokens
                    target_vocab_size = min(
                        8000,
                        base_vocab
                        + int(math.log2(max(10.0, dataset_size_mb)) * scale_factor),
                    )

                    # Ensure vocab size is reasonable (not too small, not too large)
                    target_vocab_size = max(256, min(10000, target_vocab_size))

                    # Round to nearest 100 for cleaner numbers
                    target_vocab_size = int(round(target_vocab_size / 100) * 100)

                    print("🔧 Training custom BPE tokenizer...")
                    print(
                        f"   Dataset size: {dataset_size_mb:.2f} MB, Unique chars: {num_unique_chars}"
                    )
                    print(f"   Auto-selected vocab_size: {target_vocab_size:,}")

            # Initialize tokenizer with BPE model
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()

            # Train the tokenizer (no verbose logging by default - clean and simple!)
            trainer = BpeTrainer(vocab_size=target_vocab_size, special_tokens=["<unk>"])
            tokenizer.train_from_iterator([text], trainer=trainer)

            vocab_size = tokenizer.get_vocab_size()

            def encode(s):
                return tokenizer.encode(s).ids

            def decode(token_ids):
                return tokenizer.decode(token_ids)

            print(f"✅ Custom BPE tokenizer trained (vocab_size={vocab_size:,})")
        except ImportError:
            print("❌ tokenizers not installed. Install with: pip install tokenizers")
            print("📝 Falling back to character-level tokenization")
            # Fall through to character-level implementation
            TOKENIZATION_METHOD = "character"

    if TOKENIZATION_METHOD == "character" or TOKENIZATION_METHOD not in [
        "gpt2",
        "custom_bpe",
    ]:
        # Character-level: Simple, fast, best for small datasets
        # Create vocabulary from all unique characters in the text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        # Create character-to-integer and integer-to-character mappings
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        # Encoder: convert string to list of integers
        def encode(s):
            return [stoi[c] for c in s]

        # Decoder: convert list of integers to string
        def decode(token_ids):
            return "".join([itos[i] for i in token_ids])

        print(f"📝 Using character-level tokenization (vocab_size={vocab_size})")
        print("✅ Recommended for small datasets - fastest training and lowest loss")

    # Add vocab_size to hyperparameters after it's determined (works for all tokenization methods)
    hyperparameters["vocab_size"] = vocab_size

    # Encode entire text dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split data into train and validation sets (90% train, 10% validation)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================


def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y

    Args:
        split: 'train' or 'val' to select which dataset to use

    Returns:
        x: Input sequences of shape (batch_size, block_size)
        y: Target sequences of shape (batch_size, block_size)
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss of the model on the train and validation sets

    Returns:
        out: Dictionary containing the loss on the train and validation sets
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_model_name(model_instance):
    """Extract model name from model class (e.g., 'BigramLanguageModel' -> 'bigram')"""
    if MODEL_TYPE == "gpt2":
        # For GPT-2, use the model name from the wrapper
        if hasattr(model_instance, "model_name"):
            return model_instance.model_name.replace(
                "-", ""
            )  # gpt2-medium -> gpt2medium
        return "gpt2"

    class_name = model_instance.__class__.__name__
    # Convert PascalCase to lowercase (simple heuristic: first word before 'LanguageModel')
    if "LanguageModel" in class_name:
        return class_name.replace("LanguageModel", "").lower()
    if "Wrapper" in class_name:
        return class_name.replace("Wrapper", "").lower()
    return class_name.lower()


def save_checkpoint(step, model, optimizer, model_name, source_name):
    """Save model checkpoint with metadata"""
    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparameters": hyperparameters,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "batch_size": batch_size,
    }

    # Generate checkpoint filename
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    checkpoint_name = (
        f"checkpoint_{model_name}_{source_name}_step{step:06d}_{timestamp}.pt"
    )
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    try:
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    except Exception as e:
        print(f"⚠️  Error saving checkpoint: {e}")
        return None


# ============================================================================
# TRAINING SETUP
# ============================================================================

# Initialize model based on MODEL_TYPE
if MODEL_TYPE == "gpt2":
    # GPT-2 model (pre-trained from HuggingFace)
    from models.gpt2_wrapper import GPT2Wrapper

    print(f"🤖 Using GPT-2 model: {GPT2_MODEL_NAME}")
    if USE_LORA:
        print("🔧 Using LoRA for efficient fine-tuning")
        print(
            f"   LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}"
        )

    model = GPT2Wrapper(
        model_name=GPT2_MODEL_NAME,
        use_lora=USE_LORA,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        device=device,
    )

    # Use GPT-2's tokenizer (already set up in GPT2Wrapper)
    encode = model.encode
    decode = model.decode
    vocab_size = model.get_vocab_size()

    # Encode entire text dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split data into train and validation sets (90% train, 10% validation)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # GPT-2 uses its own block_size (from config), but we'll use our batch_size
    # Note: GPT-2's max position embeddings might limit block_size
    gpt2_config = model.model.config
    max_pos = gpt2_config.max_position_embeddings
    if block_size > max_pos:
        print(
            f"⚠️  block_size ({block_size}) > GPT-2 max_pos ({max_pos}), using {max_pos}"
        )
        block_size = max_pos

    model.to(device)  # Ensure model is on device

elif MODEL_TYPE == "from_scratch":
    # Original from-scratch model (with or without LoRA)
    if USE_LORA:
        from models.bigram_lm_v2_lora import BigramLanguageModelLoRA

        print("🔧 Using LoRA for efficient fine-tuning")
        print(
            f"   LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}"
        )
        model = BigramLanguageModelLoRA(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            device=device,
            dropout=dropout,
            n_head=n_head,
            n_layer=n_layer,
            use_lora=True,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
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

    model.to(device)  # move model to device
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'from_scratch' or 'gpt2'")

# Count parameters (for both GPT-2 and from_scratch models)
if MODEL_TYPE == "gpt2":
    # GPT-2 always has parameter info method
    param_info = model.get_parameter_info()
    print("📊 Parameter Statistics:")
    print(f"   Model: {param_info.get('model_name', GPT2_MODEL_NAME)}")
    print(f"   Total parameters: {param_info['total']:,}")
    if USE_LORA:
        print(f"   Trainable (total): {param_info['trainable']:,}")
        if "lora_only_params" in param_info:
            print(
                f"     - LoRA adapters: {param_info['lora_only_params']:,} ({param_info.get('lora_only_percentage', 0):.2f}%)"
            )
        print(f"   Frozen (base model): {param_info['frozen']:,}")
        lora_pct = param_info.get(
            "lora_percentage", param_info.get("lora_only_percentage", 0)
        )
        print(f"   💰 LoRA savings: Training only {lora_pct:.2f}% of parameters!")
    else:
        print(f"   Trainable parameters: {param_info['trainable']:,}")
        print(
            "   💡 Tip: Use USE_LORA=True for efficient fine-tuning (90-99% fewer parameters)"
        )
elif USE_LORA:
    param_info = model.get_parameter_info()
    print("📊 Parameter Statistics:")
    print(f"   Total parameters: {param_info['total']:,}")
    print(f"   Trainable (total): {param_info['trainable']:,}")
    print(
        f"     - LoRA adapters: {param_info['lora_only_params']:,} ({param_info['lora_only_percentage']:.2f}%)"
    )
    print(
        f"     - Embeddings: {param_info['embedding_params']:,} ({param_info['embedding_percentage']:.2f}%)"
    )
    print(f"   Frozen (base model): {param_info['frozen']:,}")
    print(
        f"   💰 LoRA savings: Training only {param_info['lora_only_percentage']:.2f}% of parameters (LoRA-only)!"
    )
    if param_info["embedding_percentage"] > 5.0:
        print(
            f"   ⚠️  Note: Embeddings are {param_info['embedding_percentage']:.1f}% of total (higher in small models)"
        )

    # Warn about performance for small models
    if param_info["total"] < 5_000_000:  # Less than 5M parameters
        print(
            "   ⚠️  PERFORMANCE WARNING: LoRA may be SLOWER than full fine-tuning for small models!"
        )
        print("       For models < 5M params, the LoRA overhead can outweigh benefits.")
        print(
            "       Consider using full fine-tuning (USE_LORA=False) for better speed."
        )
else:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# Verify device usage
print(f"Model device: {next(model.parameters()).device}")
if device == "mps":
    print("✅ Model successfully moved to Apple Silicon GPU (MPS)")

# Compile model for better performance (PyTorch 2.0+)
# This can provide 2-3x speedup on Apple Silicon M4
# DISABLED: torch.compile for MPS is still experimental and may cause slowdowns
try:
    if device == "mps" and hasattr(torch, "compile") and False:  # Disabled for now
        print("🔧 Compiling model for Apple Silicon... (this may take a minute)")
        model = torch.compile(model, mode="default")
        print("✅ Model compiled successfully!")
    else:
        print("ℹ️  Using MPS without compilation (torch.compile disabled for MPS)")
except Exception as e:
    print(f"⚠️  Model compilation skipped: {e}")

# Initialize optimizer
# When using LoRA, only LoRA parameters are trainable (base model is frozen)
if USE_LORA:
    # Only optimize trainable parameters (LoRA adapters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    print(
        f"✅ Optimizer initialized with {len(trainable_params)} parameter groups (LoRA only)"
    )
else:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )  # AdamW optimizer

# ============================================================================
# TRAINING LOOP
# ============================================================================

# Print progress every interval_print steps
interval_print = training_steps // 10  # print every 10% of the training steps

print(f"Starting training for {training_steps} steps...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Vocabulary size: {vocab_size:,} tokens")
if ENABLE_CHECKPOINTS:
    print(
        f"Checkpoints enabled: saving every {CHECKPOINT_INTERVAL} steps to {CHECKPOINT_DIR}/"
    )
print("-" * 50)

# Get model and source names for checkpoint naming
model_name = get_model_name(model)
source_name = get_data_source_name(TRAINING_DATA_SOURCE)

start_time = time.time()

for step in range(training_steps):

    # Every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed if step > 0 else 0
        progress_pct = (step / training_steps) * 100
        print(
            f"step {step}/{training_steps} ({progress_pct:.1f}%): train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | {elapsed:.1f}s ({steps_per_sec:.2f} steps/sec)"
        )

        # Save checkpoint if enabled
        if ENABLE_CHECKPOINTS and step % CHECKPOINT_INTERVAL == 0 and step > 0:
            checkpoint_path = save_checkpoint(
                step, model, optimizer, model_name, source_name
            )
            if checkpoint_path:
                print(f"   💾 Checkpoint saved: {checkpoint_path}")

    # Print progress more frequently in production mode to show it's not hung
    elif step > 0 and step % 25 == 0:
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed
        progress_pct = (step / training_steps) * 100
        eta_seconds = (
            (training_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
        )
        eta_minutes = eta_seconds / 60
        print(
            f"step {step}/{training_steps} ({progress_pct:.1f}%) | {steps_per_sec:.2f} steps/sec | ETA: {eta_minutes:.1f}m"
        )

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Forward pass: compute predictions and loss
    logits, loss = model(xb, yb)

    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Update parameters
    optimizer.step()

print("-" * 50)
print(f"Training complete! Final loss: {loss.item():.4f}")
total_time = time.time() - start_time
print(
    f"Total training time: {format_time(total_time)} ({training_steps/total_time:.2f} steps/sec)"
)

# Save final model checkpoint
if ENABLE_CHECKPOINTS:
    final_checkpoint_path = save_checkpoint(
        training_steps, model, optimizer, model_name, source_name
    )
    if final_checkpoint_path:
        print(f"✅ Final model saved: {final_checkpoint_path}")

# ============================================================================
# GENERATION
# ============================================================================

print("\nGenerating text...")
print("=" * 50)

# Generate text starting from a null character (index 0)
# For GPT-2, we need to use the tokenizer's bos_token or a simple prompt
if MODEL_TYPE == "gpt2":
    # GPT-2 doesn't use a null token, so we'll start with an empty sequence
    # The tokenizer will handle this appropriately
    context = torch.tensor([[]], dtype=torch.long).to(device)
    if context.shape[1] == 0:
        # If empty, add a single token (usually BOS or a space)
        context = torch.tensor(
            [[model.tokenizer.bos_token_id or 0]], dtype=torch.long
        ).to(device)
    generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
    # GPT-2 generate returns the full sequence including input, so we extract new tokens
    if generated_tokens.shape[1] > context.shape[1]:
        new_tokens = generated_tokens[0, context.shape[1] :].tolist()
    else:
        new_tokens = generated_tokens[0].tolist()
    generated_text = decode(new_tokens)
else:
    # From-scratch model: start with null token
    context = torch.zeros((1, 1), dtype=torch.long).to(device)
    generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
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

    # Generate filename (reuse model_name and source_name from checkpoint section)
    filename = generate_output_filename(
        model_name=model_name,
        source_name=source_name,
        vocab_size=vocab_size,
        training_steps=training_steps,
        test_mode=TEST_MODE,
        use_lora=USE_LORA,
        lora_rank=LORA_RANK if USE_LORA else None,
        lora_alpha=LORA_ALPHA if USE_LORA else None,
        model_type=MODEL_TYPE,
        gpt2_model_name=GPT2_MODEL_NAME if MODEL_TYPE == "gpt2" else None,
    )

    # Construct full output path
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Write output file
    if write_output_file(output_path, hyperparameters, captured_output):
        print(f"\n✅ Output written to: {output_path}")
    else:
        print("\n⚠️  Failed to write output file")
