# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Environment Setup

Set up Python environment and dependencies for LLM training:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio numpy matplotlib tiktoken
pip install ruff pytest pre-commit
pre-commit install
```

For Apple Silicon Macs, use MPS acceleration:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Common Development Tasks

### Train Small GPT Model

Start training on tiny-shakespeare dataset (Karpathy style):
```bash
# Download data and start training
python train.py --compile=True --device=mps
```

### Generate Text Samples

Sample from trained model:
```bash
python sample.py --checkpoint=checkpoint.pt --num_samples=5 --max_length=200
```

### Run Single Test

```bash
pytest tests/test_model.py::TestGPT::test_forward_pass -v
```

### Linting and Formatting

Format and lint code:
```bash
ruff format .
ruff check . --fix
```

## Architecture Overview

This project implements a transformer-based language model following Andrej Karpathy's "Let's build GPT" approach:

- **`model.py`** - Core GPT transformer implementation with multi-head attention
- **`train.py`** - Training loop with gradient accumulation and learning rate scheduling  
- **`sample.py`** - Text generation and sampling utilities
- **`data.py`** - Data loading and tokenization (BPE via tiktoken)
- **`config.py`** - Model hyperparameters and training configuration

Key components:
- Multi-head self-attention with causal masking
- Layer normalization and residual connections
- Positional embeddings for sequence modeling
- AdamW optimizer with cosine learning rate decay

## Git Workflow

Follow conventional commit format:
- `feat:` for new model features or training improvements
- `fix:` for bug fixes in training or inference
- `docs:` for documentation updates
- `refactor:` for code restructuring without functionality changes

Example: `feat: add learning rate warmup to training loop`