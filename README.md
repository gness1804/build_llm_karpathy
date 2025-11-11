# Build LLM - Andrej Karpathy Tutorial

This project is a hands-on implementation of a language model following Andrej Karpathy's tutorial video on building Large Language Models (LLMs) from scratch (https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1094s). The goal is to understand the fundamental concepts behind LLMs by implementing a simple bigram character-level language model using PyTorch.

## Project Purpose

This is a learning project designed to:
- Understand the core mechanics of language models
- Learn how neural networks predict the next token in a sequence
- Grasp fundamental concepts like embeddings, loss functions, backpropagation, and optimization
- Build intuition for how modern LLMs work, starting from the simplest possible architecture

The implementation uses Shakespeare's text as training data to (ideally) generate Shakespeare-like text after training.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure training data exists:**
   The project expects a text file at `sources/shakespeare.txt`. This file should contain the Shakespeare text corpus used for training. This file exists in the directory previously noted or it can be downloaded at https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt using your favorite document download method. 

## Running the Model

To train the model and generate text, simply run:

```bash
python training.py
```

**What happens when you run it:**
1. Loads Shakespeare text and creates character-level vocabulary
2. Splits data into training (90%) and validation (10%) sets
3. Trains a bigram language model for 10,000 steps (adjustable)
4. Prints the final loss value
5. Generates and prints 300 characters (adjustable) of Shakespeare-like text (in theory...)

**Output example:**
```
----
End of training. Loss: [some value]
[Generated Shakespeare-like text...]
```

## Technical Overview

### Architecture: Bigram Language Model

This is the **simplest** form of a language model. It predicts the next character based only on the current character, using a lookup table (embedding matrix).

**Key components:**

1. **Token Embedding Table** (`nn.Embedding(vocab_size, vocab_size)`)
   - A lookup table where each character maps to prediction scores for the next character
   - This is the model's "intelligence" - what it learns during training

2. **Forward Pass**
   - Takes input tokens (characters)
   - Looks up predictions from the embedding table
   - Computes loss by comparing predictions to actual next characters

3. **Text Generation**
   - Starts with a seed token (zero/null character)
   - Predicts next token using the model
   - Samples from the probability distribution
   - Appends to sequence and repeats

4. **Training Loop**
   - Gets batches of training data
   - Computes predictions and loss
   - Backpropagates gradients (`loss.backward()`)
   - Updates parameters (`optimizer.step()`)
   - Repeats for 10,000 iterations

### Hyperparameters

- **Block size**: 8 (sequence length)
- **Batch size**: 32 (number of sequences processed in parallel)
- **Training steps**: 10,000
- **Learning rate**: 1e-3 (0.001)
- **Optimizer**: AdamW

### How It Works

```
Input: "hell"
Model looks up: 'l' -> [probability distribution for next char]
Samples: 'o'
Output: "hello"
```

The model learns which characters typically follow other characters by seeing many examples from Shakespeare's text.

## Project Structure

```
build_llm_karpathy/
├── README.md           # This file
├── training.py         # Main training script with model implementation
├── notes.md            # Detailed learning notes and explanations
├── requirements.txt    # Python dependencies
├── docs/               # Documentation (see docs/README.md for index)
│   ├── lora/          # LoRA fine-tuning documentation
│   ├── gpt2/          # GPT-2 integration documentation
│   └── *.md           # General guides
├── sources/
│   └── shakespeare.txt # Training data
└── WARP.md            # Development environment setup
```

## Key Concepts Covered

- **Tokenization**: Converting text to integers (character-level encoding)
- **Embeddings**: Representing tokens as learnable vectors
- **Batching**: Processing multiple sequences in parallel
- **Loss Function**: Cross-entropy loss for classification
- **Backpropagation**: Computing gradients (`loss.backward()`)
- **Optimization**: Updating parameters to reduce loss (`optimizer.step()`)
- **Text Generation**: Sampling from probability distributions
- **Train/Validation Split**: Evaluating model on unseen data

## Limitations

This bigram model has significant limitations:
- Only looks at one character at a time (no context beyond the immediate previous character)
- No attention mechanism
- No multi-layer architecture
- Character-level (not subword/token-level like modern LLMs)

These limitations are intentional - this is a stepping stone to understanding more sophisticated architectures like Transformers.

## Next Steps

After understanding this bigram model, typical next steps include:
1. Implementing self-attention mechanisms
2. Adding multiple layers and residual connections
3. Implementing the full Transformer architecture
4. Using more sophisticated tokenization (BPE, WordPiece)
5. Scaling up model size and training data

## Learning Resources

- **Tutorial Video**: Andrej Karpathy's LLM tutorial (link to be added)
- **notes.md**: Detailed notes with visual explanations of key concepts
- **Training Code**: `training.py` contains extensive comments explaining each step
- **Documentation**: See [docs/README.md](docs/README.md) for comprehensive guides on:
  - Fine-tuning strategies (LoRA, full fine-tuning)
  - GPT-2 integration
  - Performance optimization
  - Checkpoint management

## Credits

This implementation follows the tutorial by Andrej Karpathy. All credit for the teaching methodology and approach goes to him.

## License

This is a personal learning project. Please refer to the original tutorial (https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1094s) for any licensing information.

