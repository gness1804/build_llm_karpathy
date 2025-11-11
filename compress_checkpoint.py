"""
Compress checkpoint files by removing optimizer state.
Reduces file size by ~50% but prevents training resumption.
Useful for keeping important checkpoints in version control.

Usage:
    python3 compress_checkpoint.py path/to/checkpoint.pt
    python3 compress_checkpoint.py checkpoints/checkpoint_*.pt  # Glob pattern
"""

import torch
import os
import sys
from pathlib import Path
import shutil

def compress_checkpoint(checkpoint_path, output_path=None, keep_original=True):
    """
    Compress a checkpoint by removing optimizer state.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Where to save compressed checkpoint (default: checkpoint_compressed.pt)
        keep_original: If True, keep original file; if False, replace it
    
    Returns:
        Tuple of (original_size_mb, compressed_size_mb, savings_pct)
    """
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: File not found: {checkpoint_path}")
        return None
    
    # Get original size
    original_size = os.path.getsize(checkpoint_path)
    original_size_mb = original_size / (1024 * 1024)
    
    print(f"📦 Compressing: {checkpoint_path}")
    print(f"   Original size: {original_size_mb:.1f} MB")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    # Create compressed version (remove optimizer state)
    compressed_checkpoint = {
        "step": checkpoint.get("step"),
        "model_state_dict": checkpoint.get("model_state_dict"),
        "hyperparameters": checkpoint.get("hyperparameters"),
        "vocab_size": checkpoint.get("vocab_size"),
        "block_size": checkpoint.get("block_size"),
        "batch_size": checkpoint.get("batch_size"),
    }
    
    # Determine output path
    if output_path is None:
        if keep_original:
            # Create _compressed suffix
            base_path = checkpoint_path.rsplit(".", 1)[0]
            output_path = f"{base_path}_compressed.pt"
        else:
            output_path = checkpoint_path
    
    # Save compressed checkpoint
    try:
        torch.save(compressed_checkpoint, output_path)
        compressed_size = os.path.getsize(output_path)
        compressed_size_mb = compressed_size / (1024 * 1024)
        savings_pct = ((original_size - compressed_size) / original_size) * 100
        
        print(f"   Compressed size: {compressed_size_mb:.1f} MB")
        print(f"   Savings: {savings_pct:.1f}%")
        print(f"   ✅ Saved to: {output_path}")
        
        # Remove original if requested
        if not keep_original and output_path != checkpoint_path:
            os.remove(checkpoint_path)
            print(f"   🗑️  Removed original file")
        
        return (original_size_mb, compressed_size_mb, savings_pct)
    
    except Exception as e:
        print(f"❌ Error saving compressed checkpoint: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compress_checkpoint.py <checkpoint_path> [--replace] [--output <path>]")
        print("\nExamples:")
        print("  python3 compress_checkpoint.py checkpoints/checkpoint_*.pt")
        print("  python3 compress_checkpoint.py checkpoint.pt --replace")
        print("  python3 compress_checkpoint.py checkpoint.pt --output compressed.pt")
        sys.exit(1)
    
    checkpoint_pattern = sys.argv[1]
    keep_original = "--replace" not in sys.argv
    output_path = None
    
    # Parse --output argument
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]
    
    # Handle glob patterns
    from glob import glob
    checkpoint_files = glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"❌ No files matching pattern: {checkpoint_pattern}")
        sys.exit(1)
    
    print(f"🔍 Found {len(checkpoint_files)} checkpoint file(s)")
    print("-" * 50)
    
    total_original = 0
    total_compressed = 0
    
    for checkpoint_path in checkpoint_files:
        result = compress_checkpoint(checkpoint_path, output_path, keep_original)
        if result:
            orig, comp, pct = result
            total_original += orig
            total_compressed += comp
        print()
    
    # Summary
    if len(checkpoint_files) > 1:
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        total_savings = ((total_original - total_compressed) / total_original) * 100
        print(f"Total original: {total_original:.1f} MB")
        print(f"Total compressed: {total_compressed:.1f} MB")
        print(f"Total savings: {total_savings:.1f}%")

if __name__ == "__main__":
    main()
