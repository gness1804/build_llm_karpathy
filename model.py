"""
Bigram Language Model - Building an LLM from Scratch
Following Andrej Karpathy's tutorial
"""

import torch

from models.bigram_lm_v2 import BigramLanguageModel

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Set random seed for reproducibility
torch.manual_seed(1337)

# Model hyperparameters
block_size = 8          # Maximum context length for predictions
batch_size = 32         # How many independent sequences to process in parallel
learning_rate = 1e-3    # Learning rate for optimizer
training_steps = 5000  # Number of training iterations
eval_iters = 200        # Number of iterations to evaluate loss
n_embed = 32           # Number of embedding dimensions
dropout = 0.1          # Dropout rate for self-attention
num_heads = 4          # Number of heads for self-attention
n_layer = 4          # Number of layers for the transformer
head_size = n_embed // num_heads  # Size of each head (must satisfy: num_heads * head_size = n_embed)

device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available, otherwise use CPU

# Generation settings
max_new_tokens = 300    # Number of characters to generate

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
model = BigramLanguageModel(vocab_size, n_embed, block_size, device, dropout, num_heads, head_size, n_layer)
model.to(device) # move model to device

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ============================================================================
# TRAINING LOOP
# ============================================================================

# Print progress every interval_print steps
interval_print = training_steps // 10

print(f"Starting training for {training_steps} steps...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Vocabulary size: {vocab_size} characters")
print("-" * 50)

for step in range(training_steps):
    # Sample a batch of data
    xb, yb = get_batch('train')
    # Forward pass: compute predictions and loss
    logits, loss = model(xb, yb)
    
    loss = loss.mean() # average loss over the batch
    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Print progress every <interval_print> steps: e.g. if interval_print is 1000, print every 1000 steps
    if (step + 1) % interval_print == 0:
        print(f"Step {step + 1}/{training_steps} - Loss: {loss.item():.4f}")

# Print loss every eval_iters steps
if (step + 1) % eval_iters == 0:
    losses = estimate_loss()
    print(f"Step {step + 1}/{training_steps} - Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")

print("-" * 50)
print(f"Training complete! Final loss: {loss.item():.4f}")

# ============================================================================
# GENERATION
# ============================================================================

print("\nGenerating text...")
print("=" * 50)

# Generate text starting from a null character (index 0)
context = torch.zeros((1, 1), dtype=torch.long)
generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
generated_text = decode(generated_tokens[0].tolist())

print(generated_text)
print("=" * 50)
