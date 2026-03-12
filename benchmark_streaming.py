#!/usr/bin/env python3
"""Benchmark comparison: fastsafe_open vs fastsafe_open_streaming vs mmap.

This script measures peak VRAM usage and loading time for three strategies:
1. fastsafe_open (original) — single giant buffer, 2x peak VRAM
2. fastsafe_open_streaming (new) — chunked I/O, ~1.03x peak VRAM  
3. safetensors mmap + to(copy=False) — ComfyUI native, 1.0x peak VRAM

Usage:
    python benchmark_streaming.py --model /path/to/model.safetensors
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from typing import Dict, Tuple

import torch


def get_vram_mb(device_idx=0) -> float:
    """Get current VRAM usage from PyTorch."""
    return torch.cuda.memory_allocated(device_idx) / (1024**2)


def get_peak_vram_mb(device_idx=0) -> float:
    """Get peak VRAM usage from PyTorch."""
    return torch.cuda.max_memory_allocated(device_idx) / (1024**2)


def reset_memory():
    """Reset CUDA memory tracking."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def format_size(mb: float) -> str:
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.1f} MB"


def benchmark_fastsafe_original(model_path: str, device: str) -> Tuple[float, float, int]:
    """Load using the original fastsafe_open (single buffer)."""
    reset_memory()
    from fastsafetensors import fastsafe_open, SingleGroup
    
    start = time.time()
    with fastsafe_open(filenames=model_path, framework="pt", pg=SingleGroup(), 
                       device=device, nogds=True) as f:
        sd = {}
        for k in f.keys():
            sd[k] = f.get_tensor(k).clone()
    elapsed = time.time() - start
    peak = get_peak_vram_mb()
    n_tensors = len(sd)
    
    # Cleanup
    del sd
    reset_memory()
    return elapsed, peak, n_tensors


def benchmark_fastsafe_streaming(model_path: str, device: str, 
                                  chunk_size: int = 1*1024**3) -> Tuple[float, float, int]:
    """Load using the new chunked streaming API."""
    reset_memory()
    from fastsafetensors import fastsafe_open_streaming
    
    start = time.time()
    with fastsafe_open_streaming(filename=model_path, device=device, 
                                  nogds=True, chunk_size=chunk_size) as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    elapsed = time.time() - start
    peak = get_peak_vram_mb()
    n_tensors = len(sd)
    
    del sd
    reset_memory()
    return elapsed, peak, n_tensors


def benchmark_mmap_copy_false(model_path: str, device: str) -> Tuple[float, float, int]:
    """Load using safetensors mmap + .to(device, copy=False)."""
    reset_memory()
    import safetensors.torch
    
    start = time.time()
    sd = safetensors.torch.load_file(model_path, device="cpu")
    for k in sd:
        sd[k] = sd[k].to(device=device, copy=False)
    elapsed = time.time() - start
    peak = get_peak_vram_mb()
    n_tensors = len(sd)
    
    del sd
    reset_memory()
    return elapsed, peak, n_tensors


def main():
    parser = argparse.ArgumentParser(description="Benchmark safetensors loading strategies")
    parser.add_argument("--model", required=True, help="Path to .safetensors file")
    parser.add_argument("--device", default="cuda:0", help="Target device")
    parser.add_argument("--chunk-size-gb", type=float, default=1.0, 
                       help="Chunk size for streaming in GB")
    parser.add_argument("--skip-original", action="store_true",
                       help="Skip the original fastsafe_open (saves time/VRAM)")
    args = parser.parse_args()

    model_path = args.model
    device = args.device
    chunk_size = int(args.chunk_size_gb * 1024**3)

    file_size_mb = os.path.getsize(model_path) / (1024**2)
    print(f"Model: {model_path}")
    print(f"File size: {format_size(file_size_mb)}")
    print(f"Device: {device}")
    print(f"Chunk size: {format_size(chunk_size / (1024**2))}")
    print("=" * 80)

    results = []

    # Strategy 1: mmap + copy=False (baseline — always safe)
    print("\n[1/3] mmap + to(copy=False) ...")
    try:
        elapsed, peak, n = benchmark_mmap_copy_false(model_path, device)
        ratio = peak / file_size_mb
        results.append(("mmap + copy=False", elapsed, peak, ratio, n, "✓"))
        print(f"  Peak: {format_size(peak)} ({ratio:.2f}x), Time: {elapsed:.1f}s, Tensors: {n}")
    except Exception as e:
        results.append(("mmap + copy=False", 0, 0, 0, 0, f"FAIL: {e}"))
        print(f"  FAILED: {e}")

    # Strategy 2: fastsafe_open_streaming (new chunked approach)
    print("\n[2/3] fastsafe_open_streaming (chunked) ...")
    try:
        elapsed, peak, n = benchmark_fastsafe_streaming(model_path, device, chunk_size)
        ratio = peak / file_size_mb
        results.append(("streaming (chunked)", elapsed, peak, ratio, n, "✓"))
        print(f"  Peak: {format_size(peak)} ({ratio:.2f}x), Time: {elapsed:.1f}s, Tensors: {n}")
    except Exception as e:
        results.append(("streaming (chunked)", 0, 0, 0, 0, f"FAIL: {e}"))
        print(f"  FAILED: {e}")

    # Strategy 3: Original fastsafe_open (2x spike expected)
    if not args.skip_original:
        print("\n[3/3] fastsafe_open (original, single buffer) ...")
        try:
            elapsed, peak, n = benchmark_fastsafe_original(model_path, device)
            ratio = peak / file_size_mb
            results.append(("fastsafe_open (original)", elapsed, peak, ratio, n, "✓"))
            print(f"  Peak: {format_size(peak)} ({ratio:.2f}x), Time: {elapsed:.1f}s, Tensors: {n}")
        except Exception as e:
            results.append(("fastsafe_open (original)", 0, 0, 0, 0, f"FAIL: {e}"))
            print(f"  FAILED: {e}")
    else:
        print("\n[3/3] fastsafe_open (original) — SKIPPED")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<30} {'Peak':>10} {'Ratio':>8} {'Time':>8} {'Status'}")
    print("-" * 80)
    for name, elapsed, peak, ratio, n, status in results:
        if status == "✓":
            verdict = "PASS" if ratio < 1.1 else f"FAIL ({ratio:.2f}x)"
            print(f"{name:<30} {format_size(peak):>10} {ratio:>7.2f}x {elapsed:>7.1f}s {verdict}")
        else:
            print(f"{name:<30} {'N/A':>10} {'N/A':>8} {'N/A':>8} {status}")


if __name__ == "__main__":
    main()
