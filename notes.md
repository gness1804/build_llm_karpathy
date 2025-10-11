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