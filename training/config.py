"""
Configuration management for unified training script.

Supports both fresh training and resuming from checkpoint.
Centralizes all hyperparameter logic and env var parsing.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import os
import torch


@dataclass
class TrainingConfig:
    """Unified training configuration."""
    
    # Core training
    model_type: str  # "gpt2" or "from_scratch"
    use_lora: bool
    training_data_source: str
    batch_size: int
    block_size: int
    training_steps: int
    start_step: int = 0  # 0 for fresh, checkpoint step for resume
    learning_rate: float = 3e-4
    
    # Model architecture (for from_scratch)
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2
    
    # GPT-2 specifics
    gpt2_model_name: str = "gpt2"
    
    # Tokenization
    tokenization_method: str = "character"  # "character", "gpt2", "custom_bpe"
    custom_vocab_size: Optional[int] = None
    
    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Output
    enable_output_to_file: bool = True
    output_dir: str = "outputs"
    
    # Misc
    test_mode: bool = False
    eval_interval: int = 500
    eval_iters: int = 20
    max_new_tokens: int = 300
    generation_temperature: float = 0.7
    generation_top_k: int = 50
    device: str = "auto"
    use_lr_warmup: bool = False


def _select_device(device_env: str = "auto") -> str:
    """Select appropriate device (CUDA > MPS > CPU)."""
    if device_env != "auto":
        return device_env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config_fresh_from_env() -> TrainingConfig:
    """Build fresh training config from environment variables."""
    is_test_mode = os.environ.get("TEST_MODE", "False") == "True"
    model_type = os.environ.get("MODEL_TYPE", "from_scratch").lower()
    use_lora = os.environ.get("USE_LORA", "False").lower() == "true"
    
    device = _select_device(os.environ.get("DEVICE", "auto"))
    
    # Data
    training_data_source = os.environ.get(
        "TRAINING_DATA_SOURCE",
        "sources/training_data_final_merged.md"
    )
    
    # Determine hyperparameters based on mode and model type
    if is_test_mode:
        batch_size = int(os.environ.get("BATCH_SIZE", "32"))
        block_size = int(os.environ.get("BLOCK_SIZE", "64"))
        training_steps = int(os.environ.get("TRAINING_STEPS", "1000"))
        eval_interval = 100
        learning_rate = float(os.environ.get("LEARNING_RATE", "3e-4"))
        eval_iters = 50
        n_embd = 128
        n_head = 4
        n_layer = 3
        dropout = 0.2
    else:
        if model_type == "gpt2":
            batch_size = int(os.environ.get("BATCH_SIZE", "16"))
            block_size = int(os.environ.get("BLOCK_SIZE", "128"))
            eval_iters = 20
            learning_rate = float(os.environ.get("LEARNING_RATE", "2e-5"))
            # These are not used for GPT-2 but kept for consistency
            n_embd = 256
            n_head = 4
            n_layer = 4
        else:
            batch_size = int(os.environ.get("BATCH_SIZE", "64"))
            block_size = int(os.environ.get("BLOCK_SIZE", "128"))
            eval_iters = 50
            learning_rate = float(os.environ.get("LEARNING_RATE", "3e-4"))
            n_embd = 256
            n_head = 4
            n_layer = 4
        
        training_steps = int(os.environ.get("TRAINING_STEPS", "5000"))
        eval_interval = 500
        dropout = 0.2
    
    use_lr_warmup = os.environ.get("USE_LR_WARMUP", "True").lower() == "true"
    
    return TrainingConfig(
        model_type=model_type,
        use_lora=use_lora,
        training_data_source=training_data_source,
        batch_size=batch_size,
        block_size=block_size,
        training_steps=training_steps,
        start_step=0,
        learning_rate=learning_rate,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        gpt2_model_name=os.environ.get("GPT2_MODEL_NAME", "gpt2"),
        tokenization_method=os.environ.get("TOKENIZATION_METHOD", "character"),
        custom_vocab_size=os.environ.get("CUSTOM_VOCAB_SIZE", None),
        enable_checkpoints=os.environ.get("ENABLE_CHECKPOINTS", "False").lower() == "true",
        checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "500")),
        checkpoint_dir=os.environ.get("CHECKPOINT_DIR", "checkpoints"),
        log_dir=os.environ.get("LOG_DIR", "logs"),
        enable_output_to_file=os.environ.get("ENABLE_OUTPUT_TO_FILE", "True").lower() == "true",
        output_dir=os.environ.get("OUTPUT_DIR", "outputs"),
        test_mode=is_test_mode,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        max_new_tokens=300,
        generation_temperature=0.7,
        generation_top_k=50,
        device=device,
        use_lr_warmup=use_lr_warmup,
    )


def load_config_from_checkpoint(checkpoint: dict) -> TrainingConfig:
    """Build config from checkpoint hyperparameters + env variable overrides."""
    
    # Start with checkpoint hyperparameters
    checkpoint_hparams = checkpoint.get("hyperparameters", {})
    
    # Extract what we can from checkpoint
    checkpoint_resume_step = checkpoint.get("step", 0)
    checkpoint_block_size = checkpoint.get("block_size", 128)
    checkpoint_batch_size = checkpoint.get("batch_size", 16)
    
    # Determine model type and other settings from checkpoint
    model_type = checkpoint_hparams.get("model_type", "from_scratch").lower()
    use_lora = checkpoint_hparams.get("use_lora", False)
    tokenization_method = checkpoint_hparams.get("tokenization_method", "character")
    
    # Allow overrides from environment
    batch_size = int(os.environ.get("BATCH_SIZE", checkpoint_batch_size))
    learning_rate = float(os.environ.get("LEARNING_RATE", checkpoint_hparams.get("learning_rate", 3e-4)))
    
    # Number of additional steps to train
    resume_steps = int(os.environ.get("RESUME_STEPS", "5000"))
    
    # Training data source: use env if set, otherwise checkpoint's source
    training_data_source = os.environ.get(
        "TRAINING_DATA_SOURCE",
        checkpoint_hparams.get("training_data_source", "sources/carolyn_hax/carolyn_hax_merged_cleaned.md")
    )
    
    device = _select_device(os.environ.get("DEVICE", "auto"))
    
    return TrainingConfig(
        model_type=model_type,
        use_lora=use_lora,
        training_data_source=training_data_source,
        batch_size=batch_size,
        block_size=checkpoint_block_size,
        training_steps=resume_steps,
        start_step=checkpoint_resume_step,
        learning_rate=learning_rate,
        n_embd=checkpoint_hparams.get("n_embd", 256),
        n_head=checkpoint_hparams.get("n_head", 4),
        n_layer=checkpoint_hparams.get("n_layer", 4),
        dropout=checkpoint_hparams.get("dropout", 0.2),
        gpt2_model_name=checkpoint_hparams.get("gpt2_model_name", "gpt2"),
        tokenization_method=tokenization_method,
        custom_vocab_size=checkpoint_hparams.get("custom_vocab_size"),
        enable_checkpoints=os.environ.get("ENABLE_CHECKPOINTS", "true").lower() == "true",
        checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", checkpoint_hparams.get("checkpoint_interval", 500))),
        checkpoint_dir=os.environ.get("CHECKPOINT_DIR", "checkpoints"),
        log_dir=os.environ.get("LOG_DIR", "logs"),
        enable_output_to_file=os.environ.get("ENABLE_OUTPUT_TO_FILE", "true").lower() == "true",
        output_dir=os.environ.get("OUTPUT_DIR", "outputs"),
        test_mode=checkpoint_hparams.get("test_mode", False),
        eval_interval=checkpoint_hparams.get("eval_interval", 500),
        eval_iters=checkpoint_hparams.get("eval_iters", 20),
        max_new_tokens=checkpoint_hparams.get("max_new_tokens", 300),
        generation_temperature=checkpoint_hparams.get("generation_temperature", 0.7),
        generation_top_k=checkpoint_hparams.get("generation_top_k", 50),
        device=device,
        use_lr_warmup=checkpoint_hparams.get("use_lr_warmup", False),
    )


def config_to_dict(config: TrainingConfig) -> dict:
    """Convert config to dict for storing in checkpoint."""
    return asdict(config)
