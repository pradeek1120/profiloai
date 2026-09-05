"""Create the base synthetic profiler dataset for ProfiloAI.

The hackathon demo can run without access to AMD Developer Cloud, so this file
keeps a small but useful seed dataset in the repository. Extended scenarios are
added by ``09_more_training_data.py``.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "profiler_output": "Memory Bandwidth: 16.8%\nL2 Cache Hit Rate: 34.2%\nOccupancy: 43.1%\nKernel: attention_forward",
        "diagnosis": "CRITICAL: Memory bandwidth is only 16.8%. Root cause is likely non-coalesced reads in attention_forward. Fix by storing Q/K/V tensors contiguously, using aligned vector loads, and checking stride/layout before the kernel. Expected improvement: 1.8x-2.5x for the hot kernel.",
        "bottleneck_type": "memory_bandwidth",
        "severity": "critical",
    },
    {
        "profiler_output": "GPU Utilization: 23.0%\nCPU Utilization: 98.0%\nGPU Idle: 77.0%\nBatch Load Time: 924ms\nForward Pass: 31ms",
        "diagnosis": "CRITICAL: The GPU is waiting on the CPU data pipeline. Increase DataLoader num_workers, enable pin_memory=True, set persistent_workers=True, and prefetch batches. Expected improvement: GPU utilization can move from 23% to 70%+ if input loading is the main bottleneck.",
        "bottleneck_type": "dataloader",
        "severity": "critical",
    },
    {
        "profiler_output": "Matrix Core Utilization: 4.2%\nFP32 Operations: 94.8%\nBF16 Operations: 5.2%\nTraining Throughput: 847 tokens/sec",
        "diagnosis": "CRITICAL: The model is running mostly FP32, so MI300X Matrix Cores are underused. Enable bf16=True in TrainingArguments or torch.autocast(dtype=torch.bfloat16), and keep fp16 disabled on ROCm unless validated. Expected improvement: higher tensor throughput and lower memory use.",
        "bottleneck_type": "mixed_precision",
        "severity": "critical",
    },
    {
        "profiler_output": "Occupancy: 21.4%\nVGPR Usage: 192 registers\nShared Memory: 64KB/64KB\nActive CUs: 6/228",
        "diagnosis": "HIGH: Occupancy is limited by high VGPR and shared memory use. Reduce per-thread temporaries, split the kernel into smaller passes, and tune block size. For HIP kernels, inspect VGPR count and consider compiler limits only after simplifying the kernel. Expected improvement: 1.5x-3x depending on occupancy recovery.",
        "bottleneck_type": "low_occupancy",
        "severity": "high",
    },
    {
        "profiler_output": "Warp Execution Efficiency: 31.4%\nBranch Divergence Rate: 68.6%\nActive Lanes per Warp: 20.1/64",
        "diagnosis": "CRITICAL: Branch divergence is wasting most lanes. Replace branch-heavy per-element logic with predicated or branchless operations, or reorder data so neighboring threads take the same path. Expected improvement: warp efficiency can recover from 31% to 80%+.",
        "bottleneck_type": "warp_divergence",
        "severity": "critical",
    },
    {
        "profiler_output": "H2D Transfers: 4096 calls | avg 2.1KB\nD2H Transfers: 8192 calls | avg 1.8KB\nTransfer overhead: 38.4%",
        "diagnosis": "CRITICAL: Too many tiny CPU/GPU transfers are forcing synchronization. Avoid loss.item(), tensor.cpu(), or numpy() inside every training step. Accumulate metrics on GPU and sync every N steps; use non_blocking=True for transfers. Expected improvement: 1.3x-1.8x end-to-end.",
        "bottleneck_type": "pcie_transfer_overhead",
        "severity": "critical",
    },
    {
        "profiler_output": "Kernel Launch Overhead: 48.3%\nAverage Kernel Duration: 32 ns\nKernel Launches/sec: 2847291\nGPU Compute Utilization: 21.4%",
        "diagnosis": "CRITICAL: Work is split into too many tiny kernels. Use torch.compile(mode='reduce-overhead'), fuse scale/bias/activation operations, and batch small operations together. Expected improvement: lower launch overhead and much higher GPU utilization.",
        "bottleneck_type": "kernel_launch_overhead",
        "severity": "critical",
    },
    {
        "profiler_output": "Matrix Core Utilization: 23.1%\nMatrix dimensions: 127x255x511\nMemory Bandwidth: 61.4%\nOccupancy: 72.3%",
        "diagnosis": "HIGH: Matrix dimensions are not aligned for efficient Matrix Core use. Round hidden sizes and GEMM dimensions to multiples of 16, preferably 64 or 128. Expected improvement: Matrix Core utilization can improve from ~23% to 80%+ for aligned GEMMs.",
        "bottleneck_type": "matrix_dimension_misalignment",
        "severity": "high",
    },
]

output_file = OUTPUT_DIR / "profiler_samples.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for s in SAMPLES:
        f.write(json.dumps(s) + "\n")

print(f"Saved {len(SAMPLES)} samples to {output_file}")
