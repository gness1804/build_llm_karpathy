#!/usr/bin/env python3
"""
Quick script to check if Apple Silicon GPU (MPS) is available and working
"""

import torch
import time

print("=" * 60)
print("Apple Silicon GPU (MPS) Detection")
print("=" * 60)

# Check if MPS is available
mps_available = torch.backends.mps.is_available()
print(f"\n✓ MPS Available: {mps_available}")

if mps_available:
    print("✓ Metal Performance Shaders backend is ready!")

    # Test MPS performance
    print("\n" + "=" * 60)
    print("Running Performance Test...")
    print("=" * 60)

    # Test matrix multiplication on MPS vs CPU
    size = 1024
    iterations = 100

    # CPU test
    print(f"\nCPU Test: {iterations} iterations of {size}x{size} matrix multiplication")
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)

    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.3f}s ({iterations/cpu_time:.1f} ops/sec)")

    # MPS test
    print(f"\nMPS Test: {iterations} iterations of {size}x{size} matrix multiplication")
    x_mps = torch.randn(size, size, device="mps")
    y_mps = torch.randn(size, size, device="mps")

    # Warm up
    for _ in range(10):
        _ = torch.matmul(x_mps, y_mps)

    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x_mps, y_mps)
    mps_time = time.time() - start
    print(f"MPS Time: {mps_time:.3f}s ({iterations/mps_time:.1f} ops/sec)")

    # Speedup
    speedup = cpu_time / mps_time
    print(f"\n🚀 MPS Speedup: {speedup:.2f}x faster than CPU")

    if speedup > 2:
        print("✅ MPS is working great!")
    elif speedup > 1:
        print("⚠️  MPS is working but slower than expected")
    else:
        print("❌ MPS appears slower than CPU - something may be wrong")

else:
    print("❌ MPS is NOT available")
    print("\nPossible reasons:")
    print("  1. Not running on Apple Silicon (M1/M2/M3/M4)")
    print("  2. macOS version is too old (need macOS 12.3+)")
    print("  3. PyTorch version doesn't support MPS (need 1.12+)")

    print(f"\nPyTorch version: {torch.__version__}")
    print("To install latest PyTorch:")
    print("  pip3 install --upgrade torch torchvision torchaudio")

print("\n" + "=" * 60)
print("System Info")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Check built-in MPS
if hasattr(torch.backends.mps, "is_built"):
    print(f"MPS built-in: {torch.backends.mps.is_built()}")

print("=" * 60)
