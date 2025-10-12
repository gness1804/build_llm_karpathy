"""
Bigram Language Model - Fast Training Configuration
Smaller model for quick testing and debugging
"""

import time
import torch

from models.bigram_lm_v2 import BigramLanguageModel

# ============================================================================
# HYPERPARAMETERS - REDUCED FOR FAST TRAINING
# ============================================================================

# Set random seed for reproducibility
torch.manual_seed(1337)

# Model hyperparameters - MUCH SMALLER
batch_size = 32         # Reduced from 64
block_size = 64         # Reduced from 256 (16x less attention computation!)
training_steps = 1000   # Reduced from 5000
eval_interval = 100     # Evaluate more frequently
learning_rate = 3e-4    # Learning rate for optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available, otherwise use CPU
eval_iters = 50         # Reduced from 200
n_embd = 128            # Reduced from 384
n_head = 4              # Reduced from 6
n_layer = 3             # Reduced from 6 (half the layers!)
dropout = 0.2           # Dropout rate for self-attention

# Generation settings
max_new_tokens = 300    # Number of characters to generate

print(f"Device: {device}")
print(f"Model size: {n_layer} layers, {n_embd} embedding dims, {n_head} heads")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Load training data
with open('sources/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary from all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create character-to-integer and integer-to-character mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder: convert string to list of integers
encode = lambda s: [stoi[c] for c in s]

# Decoder: convert list of integers to string
decode = lambda l: ''.join([itos[i] for i in l])

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

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW optimizer

# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"\nStarting training for {training_steps} steps...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Vocabulary size: {vocab_size} characters")
print("-" * 50)

start_time = time.time()

for step in range(training_steps):

    # Every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed if step > 0 else 0
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed ({steps_per_sec:.2f} steps/sec)")
    
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

