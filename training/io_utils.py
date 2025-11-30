"""
I/O and output utilities for training.

Provides:
- TeeOutput: capture stdout while displaying to terminal
- Output file management
- Checkpoint naming conventions
- Generation and text utilities
"""

import os
import sys
from datetime import datetime
from io import StringIO


class TeeOutput:
    """Capture stdout while still displaying to terminal."""

    def __init__(self, *files):
        self.files = files
        self.buffer = StringIO()

    def write(self, obj):
        # Write to all files (stdout + buffer)
        for f in self.files:
            f.write(obj)
            f.flush()
        # Also write to buffer for later retrieval
        self.buffer.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def get_data_source_name(training_data_source: str) -> str:
    """Extract data source name without extension from path."""
    # Handle both relative and absolute paths
    source_path = os.path.normpath(training_data_source)
    # Get filename with extension
    filename = os.path.basename(source_path)
    # Remove extension
    source_name = os.path.splitext(filename)[0]
    return source_name


def get_model_name(model_instance) -> str:
    """Extract model name from model class instance."""
    # Import here to avoid circular dependencies
    model_type = None
    if hasattr(model_instance, 'model_type'):
        model_type = model_instance.model_type
    
    if model_type == "gpt2" or hasattr(model_instance, "model_name"):
        # For GPT-2, use the model name from the wrapper
        if hasattr(model_instance, "model_name"):
            return model_instance.model_name.replace("-", "")  # gpt2-medium -> gpt2medium
        return "gpt2"

    class_name = model_instance.__class__.__name__
    # Convert PascalCase to lowercase (simple heuristic: first word before 'LanguageModel')
    if "LanguageModel" in class_name:
        return class_name.replace("LanguageModel", "").lower()
    if "Wrapper" in class_name:
        return class_name.replace("Wrapper", "").lower()
    return class_name.lower()


def generate_output_filename(
    model_name: str,
    source_name: str,
    vocab_size: int,
    training_steps: int,
    test_mode: bool,
    use_lora: bool = False,
    lora_rank: int = None,
    lora_alpha: float = None,
    model_type: str = "from_scratch",
    gpt2_model_name: str = None,
) -> str:
    """Generate output filename with structured naming convention."""
    # Format timestamp as MMDDYYYY_HHMMSS
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

    # Construct filename components
    components = [
        "build_llm_output",
        model_name,
        source_name,
        str(vocab_size),
        str(training_steps),
        f"test={str(test_mode).lower()}",
    ]

    # Add model type information
    if model_type == "gpt2":
        if gpt2_model_name:
            components.append(f"gpt2_{gpt2_model_name}")
        else:
            components.append("gpt2")
    else:
        components.append("from_scratch")

    # Add LoRA information if used
    if use_lora:
        lora_info = f"lora_r{lora_rank}_a{lora_alpha}"
        components.append(lora_info)
    else:
        components.append("full_ft")  # full fine-tuning

    components.extend(["OUTPUT", timestamp])

    # Join with underscores (snake_case)
    filename = "_".join(components) + ".txt"
    return filename


def write_output_file(output_path: str, hyperparameters: dict, captured_output: str) -> bool:
    """Write hyperparameters and captured output to file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write hyperparameters section
            f.write("HYPERPARAMETERS\n")
            f.write("=" * 16 + "\n")
            for key, value in hyperparameters.items():
                f.write(f"{key} = {value}\n")

            f.write("\n")

            # Write output section
            f.write("OUTPUT\n")
            f.write("=" * 6 + "\n")
            f.write(captured_output)

        return True
    except Exception as e:
        print(f"⚠️  Error writing output file: {e}")
        return False


def setup_stdout_capture(enable: bool):
    """
    Set up stdout capture if enabled.
    
    Returns:
        tuple: (tee_output, original_stdout) or (None, None) if not enabled
    """
    if not enable:
        return None, None
    
    original_stdout = sys.stdout
    tee_output = TeeOutput(sys.stdout)
    sys.stdout = tee_output
    return tee_output, original_stdout


def restore_stdout(original_stdout):
    """Restore original stdout."""
    if original_stdout is not None:
        sys.stdout = original_stdout
