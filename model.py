"""
Bigram Language Model - Building an LLM from Scratch
Following Andrej Karpathy's tutorial
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Set random seed for reproducibility
torch.manual_seed(1337)

# Model hyperparameters
block_size = 8          # Maximum context length for predictions
batch_size = 32         # How many independent sequences to process in parallel
learning_rate = 1e-3    # Learning rate for optimizer
training_steps = 10000  # Number of training iterations

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
    return x, y

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class BigramLanguageModel(nn.Module):
    """
    Simple bigram language model that predicts the next character
    based only on the current character using a lookup table.
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Compute logits (raw predictions) for what could come next after each position
        
        Args:
            idx: Input tensor of shape (B, T) containing token indices
            targets: Optional target tensor of shape (B, T)
            
        Returns:
            logits: Predictions of shape (B, T, C) or (B*T, C) if targets provided
            loss: Cross-entropy loss if targets provided, None otherwise
            
        Where:
            B = batch size
            T = sequence length (tokens in a chunk) / time dimension
            C = channels (number of features/dimensions per token, equals vocab_size here)
        """
        # Get predictions from embedding table: (B, T) -> (B, T, C)
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy loss function
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens by repeatedly predicting and sampling
        
        Args:
            idx: Starting context of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            
        Returns:
            idx: Extended sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Get predictions for all positions
            logits, loss = self(idx)
            
            # Focus only on the last time step: (B, T, C) -> (B, C)
            logits = logits[:, -1, :]
            
            # Apply softmax to convert to probabilities: (B, C)
            probs = F.softmax(logits, dim=-1)
            
            # Sample one token from the distribution: (B, C) -> (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to the running sequence: (B, T) -> (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Initialize model
model = BigramLanguageModel(vocab_size)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"Starting training for {training_steps} steps...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Vocabulary size: {vocab_size} characters")
print("-" * 50)

for step in range(training_steps):
    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Forward pass: compute predictions and loss
    logits, loss = model(xb, yb)
    
    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Print progress every 1000 steps
    if (step + 1) % 1000 == 0:
        print(f"Step {step + 1}/{training_steps} - Loss: {loss.item():.4f}")

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
