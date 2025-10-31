"""
Bigram Language Model - Building an LLM from Scratch
Following Andrej Karpathy's tutorial
"""

import time
import torch
import os
is_test_mode = os.environ.get("TEST_MODE", "False")

import tiktoken
from models.bigram_lm_v2 import BigramLanguageModel

# Option: Use sentencepiece for custom BPE tokenization (smaller vocab)
# Uncomment if you want to use a custom tokenizer trained on your data
# import sentencepiece as spm

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Set to True for fast testing with smaller model, False for full training
TEST_MODE = is_test_mode == "True"
TRAINING_DATA_SOURCE = os.environ.get("TRAINING_DATA_SOURCE", "sources/shakespeare.txt")
TOKENIZATION_METHOD = os.environ.get("TOKENIZATION_METHOD", "character")  # "character", "gpt2", or "custom_bpe"
CUSTOM_VOCAB_SIZE = os.environ.get("CUSTOM_VOCAB_SIZE", None)


# Set random seed for reproducibility
torch.manual_seed(1337)

# Model hyperparameters
if TEST_MODE:
    # Fast configuration for testing and debugging
    batch_size = 32         # Reduced from 64
    block_size = 64         # Reduced from 256 (16x less attention computation!)
    training_steps = 1000   # Reduced from 5000
    eval_interval = 100     # Evaluate more frequently
    learning_rate = 3e-4    # Learning rate for optimizer
    eval_iters = 50         # Reduced from 200
    n_embd = 128            # Reduced from 384
    n_head = 4              # Reduced from 6
    n_layer = 3             # Reduced from 6 (half the layers!)
    dropout = 0.2           # Dropout rate for self-attention
    print("🔬 TEST MODE: Using reduced hyperparameters for fast training")
else:
    # Full configuration for production training (aggressively optimized for Apple Silicon)
    batch_size = 32         # Reduced from 64 for better M4 performance
    block_size = 64         # Further reduced from 128 (4x less attention computation)
    training_steps = 5000   # Number of training iterations 
    eval_interval = 100     # More frequent feedback (reduced from 500)
    learning_rate = 3e-4    # Learning rate for optimizer
    eval_iters = 25         # Further reduced from 50 for faster eval
    n_embd = 192            # Further reduced from 256 for Apple Silicon
    n_head = 3              # Further reduced from 4 for Apple Silicon
    n_layer = 3             # Further reduced from 4 for Apple Silicon
    dropout = 0.2           # Dropout rate for self-attention
    print("🚀 FULL MODE: Using production hyperparameters (aggressively optimized for M4)")

# Device selection: prioritize MPS (Apple Silicon GPU) > CUDA > CPU
if torch.cuda.is_available():
    device = 'cuda'
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
else:
    device = 'cpu'
    print("⚠️  Using CPU (slow) - consider using MPS if available")

# Generation settings
max_new_tokens = 300    # Number of characters to generate

print(f"Device: {device}")
print(f"Model size: {n_layer} layers, {n_embd} embedding dims, {n_head} heads")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Load training data
with open(TRAINING_DATA_SOURCE, 'r', encoding='utf-8') as f:
    text = f.read()

# TOKENIZATION OPTIONS:
# Option 1: Character-level (original - fastest, simplest, best for small datasets)
# Option 2: GPT-2 BPE via tiktoken (industry standard but large vocab: 50,257 tokens)
# Option 3: Custom BPE tokenizer (balanced - train on your dataset for optimal vocab size)

# Choose tokenization method:

if TOKENIZATION_METHOD == "gpt2":
    # GPT-2 BPE: Industry standard but large vocabulary (50,257 tokens)
    # Better for large datasets, but slower and higher loss for small datasets
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    def encode(s): return enc.encode(s)
    def decode(token_ids): return enc.decode(token_ids)
    print(f"📝 Using GPT-2 BPE tokenization (vocab_size={vocab_size:,})")
    print("⚠️  Note: Large vocab may cause higher loss and slower training")
    
elif TOKENIZATION_METHOD == "custom_bpe":
    # Custom BPE: Train a tokenizer on your dataset (recommended for balanced approach)
    # This requires sentencepiece library: pip install sentencepiece
    try:
        import sentencepiece as spm
        import tempfile
        import math
        
        # Automatically scale vocab size based on dataset characteristics
        # This balances efficiency (compression) with model size (parameters)
        # Can be overridden with CUSTOM_VOCAB_SIZE environment variable
        manual_vocab_size = CUSTOM_VOCAB_SIZE
        
        dataset_size_bytes = len(text.encode('utf-8'))
        dataset_size_mb = dataset_size_bytes / (1024 * 1024)
        num_unique_chars = len(set(text))

        if manual_vocab_size and manual_vocab_size.lower() != 'none':
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
                target_vocab_size = base_vocab + int(math.log2(max(0.1, dataset_size_mb)) * scale_factor)
            elif dataset_size_mb < 10.0:
                # Medium datasets: 1000-3000 tokens
                target_vocab_size = base_vocab + int(math.log2(max(1.0, dataset_size_mb)) * scale_factor)
            else:
                # Large datasets: 3000-8000 tokens
                target_vocab_size = min(8000, base_vocab + int(math.log2(max(10.0, dataset_size_mb)) * scale_factor))
            
                # Ensure vocab size is reasonable (not too small, not too large)
                target_vocab_size = max(256, min(10000, target_vocab_size))
                
                # Round to nearest 100 for cleaner numbers
                target_vocab_size = int(round(target_vocab_size / 100) * 100)
                
                print("🔧 Training custom BPE tokenizer...")
                print(f"   Dataset size: {dataset_size_mb:.2f} MB, Unique chars: {num_unique_chars}")
                print(f"   Auto-selected vocab_size: {target_vocab_size:,}")
        
        # Save text to temporary file for sentencepiece training
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_file = f.name
        
        # Train the tokenizer
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix='custom_tokenizer',
            vocab_size=target_vocab_size,
            character_coverage=0.9995,
            model_type='bpe'
        )
        
        # Load the trained tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load('custom_tokenizer.model')
        vocab_size = len(sp)
        
        def encode(s): return sp.encode_as_ids(s)
        def decode(token_ids): return sp.decode_ids(token_ids)
        
        # Cleanup
        import os
        os.unlink(temp_file)
        os.unlink('custom_tokenizer.model')
        os.unlink('custom_tokenizer.vocab')
        
        print(f"✅ Custom BPE tokenizer trained (vocab_size={vocab_size:,})")
    except ImportError:
        print("❌ sentencepiece not installed. Install with: pip install sentencepiece")
        print("📝 Falling back to character-level tokenization")
        # Fall through to character-level implementation
        TOKENIZATION_METHOD = "character"
    
if TOKENIZATION_METHOD == "character" or TOKENIZATION_METHOD not in ["gpt2", "custom_bpe"]:
    # Character-level: Simple, fast, best for small datasets
    # Create vocabulary from all unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create character-to-integer and integer-to-character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encoder: convert string to list of integers
    def encode(s): return [stoi[c] for c in s]
    
    # Decoder: convert list of integers to string
    def decode(token_ids): return ''.join([itos[i] for i in token_ids])
    print(f"📝 Using character-level tokenization (vocab_size={vocab_size})")
    print("✅ Recommended for small datasets - fastest training and lowest loss")

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
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Initialize model
model = BigramLanguageModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, device=device, dropout=dropout, n_head=n_head, n_layer=n_layer)
model.to(device) # move model to device

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# Verify device usage
print(f"Model device: {next(model.parameters()).device}")
if device == 'mps':
    print("✅ Model successfully moved to Apple Silicon GPU (MPS)")

# Compile model for better performance (PyTorch 2.0+)
# This can provide 2-3x speedup on Apple Silicon M4
# DISABLED: torch.compile for MPS is still experimental and may cause slowdowns
try:
    if device == 'mps' and hasattr(torch, 'compile') and False:  # Disabled for now
        print("🔧 Compiling model for Apple Silicon... (this may take a minute)")
        model = torch.compile(model, mode='default')
        print("✅ Model compiled successfully!")
    else:
        print("ℹ️  Using MPS without compilation (torch.compile disabled for MPS)")
except Exception as e:
    print(f"⚠️  Model compilation skipped: {e}")

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW optimizer

# ============================================================================
# TRAINING LOOP
# ============================================================================

# Print progress every interval_print steps
interval_print = training_steps // 10 # print every 10% of the training steps

print(f"Starting training for {training_steps} steps...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Vocabulary size: {vocab_size:,} tokens")
print("-" * 50)

start_time = time.time()

for step in range(training_steps):

    # Every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed if step > 0 else 0
        progress_pct = (step / training_steps) * 100
        print(f"step {step}/{training_steps} ({progress_pct:.1f}%): train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | {elapsed:.1f}s ({steps_per_sec:.2f} steps/sec)")
    
    # Print progress more frequently in production mode to show it's not hung
    elif step > 0 and step % 25 == 0:
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed
        progress_pct = (step / training_steps) * 100
        eta_seconds = (training_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
        eta_minutes = eta_seconds / 60
        print(f"step {step}/{training_steps} ({progress_pct:.1f}%) | {steps_per_sec:.2f} steps/sec | ETA: {eta_minutes:.1f}m")
    
    # Sample a batch of data
    xb, yb = get_batch('train')

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
print(f"Total training time: {total_time:.1f}s ({training_steps/total_time:.2f} steps/sec)")

# ============================================================================
# GENERATION
# ============================================================================

print("\nGenerating text...")
print("=" * 50)

# Generate text starting from a null character (index 0)
context = torch.zeros((1, 1), dtype=torch.long).to(device)
generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
generated_text = decode(generated_tokens[0].tolist())

print(generated_text)
print("=" * 50)
