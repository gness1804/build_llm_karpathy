"""
Checkpoint management for training.

Provides:
- Saving and loading checkpoint files
- Checkpoint naming conventions
- Checkpoint log file management
"""

import os
import torch
from datetime import datetime
from typing import Optional, Dict, Any


def save_checkpoint(
    step: int,
    model,
    optimizer,
    config,
    vocab_size: int,
    block_size: int,
    batch_size: int,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "model",
    source_name: str = "data",
) -> Optional[str]:
    """
    Save a training checkpoint.
    
    Args:
        step: Current training step
        model: Model instance
        optimizer: Optimizer instance
        config: TrainingConfig instance
        vocab_size: Vocabulary size
        block_size: Block size used in training
        batch_size: Batch size used in training
        checkpoint_dir: Directory to save checkpoint
        model_name: Name of model (for filename)
        source_name: Name of data source (for filename)
    
    Returns:
        Path to saved checkpoint, or None if save failed
    """
    from training.config import config_to_dict
    
    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparameters": config_to_dict(config),
        "vocab_size": vocab_size,
        "block_size": block_size,
        "batch_size": batch_size,
    }

    # Generate checkpoint filename
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    checkpoint_name = (
        f"checkpoint_{model_name}_{source_name}_step{step:06d}_{timestamp}.pt"
    )
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    except Exception as e:
        print(f"⚠️  Error saving checkpoint: {e}")
        return None


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint onto
    
    Returns:
        Dictionary containing checkpoint data
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is corrupted
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")


def get_checkpoint_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract useful information from a checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
    
    Returns:
        Dictionary with checkpoint metadata
    """
    return {
        "step": checkpoint.get("step", 0),
        "vocab_size": checkpoint.get("vocab_size"),
        "block_size": checkpoint.get("block_size"),
        "batch_size": checkpoint.get("batch_size"),
        "model_type": checkpoint.get("hyperparameters", {}).get("model_type"),
        "use_lora": checkpoint.get("hyperparameters", {}).get("use_lora"),
        "training_data_source": checkpoint.get("hyperparameters", {}).get("training_data_source"),
    }


def create_checkpoint_log_file(
    log_dir: str,
    model_name: str,
    source_name: str,
    is_resume: bool = False,
    resume_step: int = 0,
) -> tuple:
    """
    Create and open a checkpoint log file.
    
    Args:
        log_dir: Directory for logs
        model_name: Name of model
        source_name: Name of data source
        is_resume: Whether this is a resume run
        resume_step: Step being resumed from (if resume)
    
    Returns:
        tuple: (log_file_handle, log_file_path) or (None, None) if not enabled
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    
    if is_resume:
        log_filename = f"resume_training_log_{model_name}_{source_name}_{timestamp}.log"
        header = f"Resume Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Resuming from step {resume_step}\n"
    else:
        log_filename = f"training_log_{model_name}_{source_name}_{timestamp}.log"
        header = f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    log_path = os.path.join(log_dir, log_filename)
    
    try:
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(header)
        log_file.write("=" * 80 + "\n")
        log_file.flush()
        return log_file, log_path
    except Exception as e:
        print(f"⚠️  Error creating checkpoint log: {e}")
        return None, None


def write_checkpoint_log(
    log_file,
    step: int,
    train_loss: float,
    val_loss: float,
    elapsed: float,
    steps_per_sec: float,
    tee_output=None,
):
    """
    Write checkpoint information to log file.
    
    Args:
        log_file: Open file handle for log
        step: Training step
        train_loss: Training loss
        val_loss: Validation loss
        elapsed: Elapsed time in seconds
        steps_per_sec: Steps per second
        tee_output: Optional TeeOutput instance for capturing full output
    """
    if log_file is None:
        return
    
    try:
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"CHECKPOINT at step {step} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*80}\n")
        
        if tee_output:
            log_content = tee_output.getvalue()
        else:
            log_content = f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}\n"
            log_content += f"Elapsed time: {elapsed:.1f}s\n"
            log_content += f"Steps/sec: {steps_per_sec:.2f}\n"
        
        log_file.write(log_content)
        log_file.flush()
    except Exception as e:
        print(f"⚠️  Error writing checkpoint log: {e}")


def close_checkpoint_log(log_file, log_path: str = None):
    """Close checkpoint log file."""
    if log_file is not None:
        try:
            log_file.close()
            if log_path:
                print(f"📝 Checkpoint log saved: {log_path}")
        except Exception as e:
            print(f"⚠️  Error closing checkpoint log: {e}")
