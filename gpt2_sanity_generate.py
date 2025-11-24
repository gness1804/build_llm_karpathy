"""\nSimple sanity-check script for GPT-2.

This script:
- Selects an available device (MPS, CUDA, or CPU)
- Loads a pretrained GPT-2 via GPT2Wrapper (no LoRA)
- Generates a short sample of text BEFORE any fine-tuning

Usage (from project root):
    python3 gpt2_sanity_generate.py
"""

import torch

from models.gpt2_wrapper import GPT2Wrapper


def get_device() -> str:
    """Select the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    # Load base GPT-2 (no LoRA, no fine-tuning)
    model = GPT2Wrapper(model_name="gpt2", use_lora=False, device=device)

    # Build an initial context: start from BOS token if available,
    # otherwise fall back to a single EOS/zero token
    tokenizer = model.tokenizer
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        # GPT-2 often uses eos_token as a reasonable start when no BOS exists
        bos_token_id = tokenizer.eos_token_id or 0

    context = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)

    max_new_tokens = 200
    temperature = 0.7
    top_k = 50

    print("\nGenerating text from base GPT-2 (no fine-tuning)...")
    print("=" * 50)

    generated = model.generate(
        input_ids=context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
    )

    # Decode only the newly generated tokens (exclude the initial BOS token)
    new_tokens = generated[0, context.shape[1] :].tolist()
    text = model.decode(new_tokens)

    print(text)
    print("=" * 50)


if __name__ == "__main__":
    main()
