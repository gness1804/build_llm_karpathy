# Notes on the Video and Exercise 

This file contains notes that I have taken, often with the help of Cursor AI, to better understand how to build a large language model.

## Visual: Understanding `targets.view(B*T)` (Line 71)

```
BEFORE: targets.view(B*T)
================================
targets shape: (B, T) = (4, 8)

Batch 0 → [t₀, t₁, t₂, t₃, t₄, t₅, t₆, t₇]
Batch 1 → [t₈, t₉, t₁₀, t₁₁, t₁₂, t₁₃, t₁₄, t₁₅]
Batch 2 → [t₁₆, t₁₇, t₁₈, t₁₉, t₂₀, t₂₁, t₂₂, t₂₃]
Batch 3 → [t₂₄, t₂₅, t₂₆, t₂₇, t₂₈, t₂₉, t₃₀, t₃₁]

    ↓  .view(B*T)  ↓
    
AFTER: targets.view(B*T)
================================
targets shape: (32,)

[t₀, t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈, t₉, t₁₀, t₁₁, t₁₂, t₁₃, t₁₄, t₁₅, 
 t₁₆, t₁₇, t₁₈, t₁₉, t₂₀, t₂₁, t₂₂, t₂₃, t₂₄, t₂₅, t₂₆, t₂₇, t₂₈, t₂₉, t₃₀, t₃₁]
```

**What's happening:**
- **Before**: A 2D tensor with 4 batches (rows) × 8 tokens (columns) = 32 total elements
- **After**: A 1D tensor with all 32 elements flattened in row-major order

**Why it's needed:**
This matches the reshaping done to `logits` on line 70. PyTorch's `F.cross_entropy()` expects:
- `logits`: shape `(N, C)` where N = number of samples, C = number of classes
- `targets`: shape `(N,)` where each element is a class index

So both tensors are flattened to treat each token position across all batches as an independent prediction, going from `(4, 8)` → `(32)` for targets and `(4, 8, vocab_size)` → `(32, vocab_size)` for logits. 

## Question to Gemini: "Can you explain more what he means by 'channels'?"

Response: 
> In the video, when the speaker talks about "channels" (e.g., at 23:53, 1:01:50), they are referring to the dimensionality of the feature vector for each token or position in the sequence.

> Imagine each character or token in your input text as a single "item." When this item goes into the neural network, it's not just a single number; the network learns to represent it with a vector of numbers. Each number in that vector is a "channel" of information or a feature.

> For example, if you have a tensor of shape (Batch, Time, Channels):

> Batch (B): Represents how many independent sequences you are processing at once.
> Time (T): Represents the length of each sequence (e.g., how many characters or tokens are in a chunk).
> Channels (C): Represents the number of features or dimensions that describe each token at each position. This is the "information" held by that token.
> So, if a token has 32 channels, it means it's represented by a vector of 32 numbers, each holding a different piece of information about that token. The neural network learns what kind of information to store in these channels to make good predictions.

## Understanding the Text Generation Line (Line 97)

```python
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```

**Breaking it down from inside out:**

### 1. `torch.zeros((1, 1), dtype=torch.long)`
Creates a tensor with shape `(1, 1)` filled with zeros:
```
[[0]]
```
- Batch size = 1 (one sequence)
- Sequence length = 1 (one token)
- Value = 0 (which represents whatever character maps to index 0 in your vocabulary)

This is the "seed" or starting point for text generation.

### 2. `m.generate(idx = ..., max_new_tokens=100)`
Calls the `generate` method (lines 76-89) which:
- Takes that initial `[[0]]` token
- Generates 100 new tokens one at a time
- Returns a tensor of shape `(1, 101)` - the original token plus 100 new ones

### 3. `[0]`
Since the batch size is 1, this extracts the first (and only) sequence:
- Before: `(1, 101)` tensor
- After: `(101,)` tensor - just the sequence of 101 token indices

### 4. `.tolist()`
Converts the PyTorch tensor to a Python list:
- Before: `tensor([0, 23, 45, 12, ...])`
- After: `[0, 23, 45, 12, ...]`

### 5. `decode(...)`
Converts the list of integer indices back to text using the character mapping (defined on line 19)

### 6. `print(...)`
Displays the generated text

**In summary:** This line generates 100 characters of text starting from a single zero token, using the trained bigram language model, then converts and prints the result as human-readable text.

## How the Model Learns: `loss.backward()` and `optimizer.step()`

These two lines are what actually **improve/train** the model:

### `loss.backward()`
This **calculates the gradients** - it figures out:
- "How much did each parameter contribute to the loss?"
- "If I change parameter X by a tiny amount, how would the loss change?"

It doesn't actually change anything yet - it just computes the directions and magnitudes of change needed. This is **backpropagation**.

### `optimizer.step()`
This **updates the parameters** using those gradients:
- Takes the gradients computed by `backward()`
- Adjusts each parameter in the direction that reduces loss
- The learning rate (`lr=1e-3` from line 98) controls how big each adjustment is

**So the two-step process is:**
1. **`loss.backward()`**: "Calculate *how* to improve" (compute gradients)
2. **`optimizer.step()`**: "Actually improve" (update parameters based on gradients)

Both steps are about the parameters:
1. First: compute the gradient for each parameter
2. Second: use those gradients to update the parameters

**The complete training loop** would look something like:
```python
for iteration in range(many_iterations):
    # 1. Get data
    xb, yb = get_batch('train')
    
    # 2. Make predictions
    logits, loss = m(xb, yb)
    
    # 3. Reset gradients from previous iteration
    optimizer.zero_grad()
    
    # 4. Calculate how to improve (gradients)
    loss.backward()
    
    # 5. Actually improve (update parameters)
    optimizer.step()
```

Over many iterations, the model's `token_embedding_table` gradually learns better predictions!