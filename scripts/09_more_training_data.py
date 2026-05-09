"""
SCRIPT 9: Extended Synthetic Training Data
==========================================
Adds 10+ more bottleneck scenarios:
- Warp divergence
- Atomic collisions
- PCIe transfer overhead
- FP32 / no mixed precision
- Matrix dimension misalignment
- Kernel launch overhead
- Register spilling
- Multi-GPU all-reduce
- Cache thrashing
- Gradient checkpointing overhead

Usage:
    python scripts/09_more_training_data.py

Output:
    data/raw/extended_samples.jsonl
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are ProfiloAI, an expert AMD GPU performance engineer.
Analyze the profiler output and give specific diagnosis with exact code fix."""

EXTENDED_SAMPLES = [
    {
        "profiler_output": """
ROCm Profiler - Sparse Attention Kernel
========================================
KernelName              | Calls | TotalDurationNs
------------------------|-------|----------------
sparse_attention_kernel | 2048  | 78291047832

Hardware Metrics:
  Warp Execution Efficiency: 31.4%
  Branch Divergence Rate: 68.6%
  Active Lanes per Warp: 20.1 / 64
  Memory Bandwidth: 42.3%
""",
        "diagnosis": """🔴 CRITICAL: Warp Divergence at 68.6%

Only 20 of 64 lanes per warp are computing. The rest idle due to branching.

ROOT CAUSE: if/else inside kernel causes threads in same warp to split paths.

THE FIX — Use branchless ternary:
// Bad:
if (mask[i] > 0.0f) { result[i] = dot(q[i], k[i]); }
else { result[i] = 0.0f; }

// Good (no divergence):
result[i] = mask[i] > 0.0f ? dot(q[i], k[i]) : 0.0f;

EXPECTED IMPROVEMENT: Warp efficiency 31% → 85% | Speed 2.7x faster""",
        "bottleneck_type": "warp_divergence",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - Gradient Accumulation
=======================================
KernelName              | Calls | TotalDurationNs
------------------------|-------|----------------
atomic_add_kernel       | 8192  | 88291047832

Hardware Metrics:
  Atomic Collision Rate: 84.3%
  Memory Bandwidth: 28.1%
  L2 Cache Hit Rate: 22.1%
""",
        "diagnosis": """🔴 CRITICAL: Atomic Collision Rate 84.3%

Threads fighting over same gradient memory locations — serializing all updates.

ROOT CAUSE: Multiple threads doing atomicAdd() to same embedding indices.

THE FIX — Sort indices first, then reduce:
# Bad: naive atomicAdd (84% collision)

# Good: sort then scatter_add
sorted_idx, order = torch.sort(indices)
sorted_grad = grad_output[order]
grad = torch.zeros(vocab_size, dim)
grad.scatter_add_(0, sorted_idx.unsqueeze(1).expand_as(sorted_grad), sorted_grad)

# Or simply use sparse gradients:
embedding = nn.Embedding(vocab_size, dim, sparse=True)

EXPECTED IMPROVEMENT: Collision 84% → 8% | Speed 6x faster on gradient kernel""",
        "bottleneck_type": "atomic_collisions",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - Training Loop
===============================
Data Transfer Metrics:
  Host→Device transfers: 4096 calls | 847ms total
  Device→Host transfers: 8192 calls | 1847ms total
  PCIe Bandwidth Used: 8.2 GB/s / 64 GB/s (12.8%)
  Transfer overhead: 38.4% of total training time
  Average transfer size: 2.1 KB (very small!)
""",
        "diagnosis": """🔴 CRITICAL: PCIe Transfer Overhead = 38.4% of Training Time

12,288 tiny CPU↔GPU transfers averaging 2.1KB each.
Each transfer has fixed overhead regardless of size — you're paying full cost for tiny payloads.

ROOT CAUSE: Calling .item(), .numpy(), or .cpu() inside training loop.

THE FIX:
# Bad — forces GPU→CPU sync every step:
loss_val = loss.item()          # sync every step!

# Good — accumulate on GPU, sync rarely:
total_loss += loss.detach()     # stays on GPU
if step % 100 == 0:
    print(total_loss.item() / 100)  # sync once per 100 steps

# Also use non-blocking transfers:
tensor = tensor.to(device, non_blocking=True)

EXPECTED IMPROVEMENT: Transfer overhead 38% → 3% | Speed 1.6x faster""",
        "bottleneck_type": "pcie_transfer_overhead",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - Transformer Training
======================================
Hardware Metrics:
  Tensor Core Utilization: 4.2%
  FP32 Operations: 94.8%
  BF16 Operations: 5.2%
  GPU Memory Used: 176GB / 192GB
  Training Throughput: 847 tokens/sec
""",
        "diagnosis": """🔴 CRITICAL: Tensor Cores at 4.2% — Running in FP32

AMD MI300X Matrix Cores give:
  FP32:  48 TFLOPS
  BF16:  383 TFLOPS  ← 8x more

You are leaving 94.8% of compute on the table.

THE FIX — Enable BF16 (NOT fp16 on AMD):
from transformers import TrainingArguments

args = TrainingArguments(
    bf16=True,    # ← enable this
    fp16=False,   # ← keep False on ROCm
)

# Or native PyTorch:
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)

WHY BF16 NOT FP16: MI300X Matrix Cores are optimized for BF16.
No loss scaling needed. Better numerical stability.

EXPECTED IMPROVEMENT: 847 → ~3,200 tokens/sec | Memory 176GB → ~94GB""",
        "bottleneck_type": "fp32_no_mixed_precision",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - GEMM Operations
=================================
Hardware Metrics:
  Matrix Core Utilization: 23.1%
  Memory Bandwidth: 61.4%
  Occupancy: 72.3%
  Matrix dimensions detected: 127x255x511 (non-aligned)
""",
        "diagnosis": """🟡 HIGH: Matrix Core Utilization 23.1% — Dimension Misalignment

AMD MI300X Matrix Cores require dimensions as multiples of 16 (ideally 64 or 128).

127, 255, 511 are NOT multiples of 16 → Matrix Cores run at ~23% efficiency.

THE FIX — Round up dimensions:
# Bad:
hidden_size = 127   # not aligned

# Good:
hidden_size = 128   # aligned to 128 → full Matrix Core efficiency

# Check your model for misaligned dims:
for name, param in model.named_parameters():
    for dim in param.shape:
        if dim % 16 != 0:
            print(f"⚠️ Misaligned: {name} {param.shape}")

ALIGNMENT GUIDE:
  Minimum: multiple of 16
  Good:    multiple of 64
  Optimal: multiple of 128

EXPECTED IMPROVEMENT: Matrix Core util 23% → 89% | GEMM speed 3.8x faster""",
        "bottleneck_type": "matrix_dimension_misalignment",
        "severity": "high",
    },
    {
        "profiler_output": """
ROCm Profiler - Inference Pipeline
====================================
Hardware Metrics:
  Kernel Launch Overhead: 48.3% of total time
  Average Kernel Duration: 32 ns (extremely short!)
  Total Kernel Launches: 2,847,291 per second
  GPU Compute Utilization: 21.4%
""",
        "diagnosis": """🔴 CRITICAL: Kernel Launch Overhead = 48.3%

2.8 million kernel launches per second, each running only 32 nanoseconds.
Launch overhead (~5-10 microseconds) is 150x longer than the kernel itself.
GPU spends more time starting work than doing work.

ROOT CAUSE: Too many tiny separate kernels — scale, bias, activation each separate.

THE FIX — Use torch.compile() to auto-fuse:
# Bad: 3 separate launches
x = x * scale    # kernel 1
x = x + bias     # kernel 2
x = torch.relu(x) # kernel 3

# Good: torch.compile fuses them automatically
model = torch.compile(model, mode="reduce-overhead")

# For ROCm specifically:
model = torch.compile(model, backend="inductor", mode="reduce-overhead")

EXPECTED IMPROVEMENT: Launch overhead 48% → 4% | GPU utilization 21% → 87% | Speed 3x""",
        "bottleneck_type": "kernel_launch_overhead",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - Custom Attention Kernel
========================================
Hardware Metrics:
  VGPR Usage: 248 registers (max: 256)
  Register Spill Stores: 847,291 per kernel call
  Register Spill Loads: 923,847 per kernel call
  Scratch Memory Used: 4.2 GB
  Occupancy: 12.4%
""",
        "diagnosis": """🔴 CRITICAL: Register Spilling — 1.7M Spill Ops Per Kernel

248/256 registers used → GPU must spill to slow scratch memory.
1.7 million extra memory ops per kernel call.
High bandwidth shown is from scratch, not useful compute.

THE FIX 1 — Split kernel into two passes:
// Bad: one massive kernel needing 248 registers
// Good: split into qk_kernel (64 regs) + softmax_v_kernel (64 regs)

THE FIX 2 — Limit registers with compiler hint:
hipcc --amdgpu-num-vgpr=128 kernel.hip -o output

// Or in code:
__attribute__((amdgpu_num_vgpr(128)))
__global__ void my_kernel() { ... }

THE FIX 3 — Use shared memory for large arrays:
__shared__ float intermediate[256];  // shared, not register

EXPECTED IMPROVEMENT: Spills 1.7M → 0 | Occupancy 12% → 65% | Speed 4x""",
        "bottleneck_type": "register_spilling",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - Multi-GPU Training (4x MI300X)
================================================
KernelName          | TotalDurationNs
--------------------|----------------
all_reduce_kernel   | 98291047832

Hardware Metrics:
  All-Reduce Time: 62.4% of total training time
  GPU Compute Utilization: 31.2% (avg 4 GPUs)
  Inter-GPU Bandwidth Used: 4.2 GB/s / 896 GB/s (0.47%)
""",
        "diagnosis": """🔴 CRITICAL: All-Reduce = 62.4% of Training Time

4 GPUs should give ~3.5x speedup. You're getting ~1.4x.
Each GPU waits 62% of the time for gradient sync.

THE FIX 1 — DeepSpeed ZeRO with overlap:
args = TrainingArguments(
    deepspeed={
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,       # ← overlap comm with compute
            "reduce_scatter": True,
            "allgather_partitions": True,
        },
        "bf16": {"enabled": True},
    }
)

THE FIX 2 — Gradient accumulation reduces sync frequency:
args = TrainingArguments(
    gradient_accumulation_steps=16,  # sync 16x less often
)

EXPECTED IMPROVEMENT: All-reduce overhead 62% → 8% | GPU util 31% → 86% | Scaling 40% → 91%""",
        "bottleneck_type": "multi_gpu_allreduce",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - Large Embedding Model
=======================================
Hardware Metrics:
  L2 Cache Hit Rate: 3.2%
  L1 Cache Hit Rate: 1.8%
  Embedding Table Size: 47GB
  Vocabulary Size: 500,000
  Memory Bandwidth: 71.4%
""",
        "diagnosis": """🔴 CRITICAL: L2 Cache Hit Rate 3.2% — Cache Thrashing

47GB embedding table vs 32MB L2 cache = 0.07% fits in cache.
Every lookup goes straight to HBM3. No cache reuse at all.

THE FIX 1 — Quantize embedding table (47GB → 12GB):
from torch.ao.quantization import quantize_dynamic

model = quantize_dynamic(model, {nn.Embedding}, dtype=torch.qint8)
# 4x smaller → 4x better cache utilization

THE FIX 2 — Sort indices for locality:
sorted_idx, order = torch.sort(indices)
# Sequential access → cache-friendly

THE FIX 3 — Sparse embeddings:
embedding = nn.Embedding(vocab_size, dim, sparse=True)
# Sparse updates reduce memory pressure

EXPECTED IMPROVEMENT: Cache hit 3% → 28% | Lookup speed 2.8x | Memory 47GB → 12GB""",
        "bottleneck_type": "cache_thrashing_embeddings",
        "severity": "critical",
    },
    {
        "profiler_output": """
ROCm Profiler - Large Model Training
======================================
Hardware Metrics:
  GPU Memory Used: 191.8GB / 192GB (99.9%)
  Recomputation Overhead: 41.2% of training time
  Gradient Checkpointing: ALL layers enabled
  Batch Size: 1
  Sequence Length: 32768
""",
        "diagnosis": """🟡 HIGH: Recomputation Overhead 41% + Memory at 99.9%

Checkpointing ALL layers forces recomputing 41% of forward pass during backward.
Too aggressive — you're trading too much compute for memory.

THE FIX 1 — Selective checkpointing (not all layers):
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        x = checkpoint(self.attention, x)  # checkpoint expensive ops
        x = self.layer_norm(x)             # don't checkpoint cheap ops
        x = checkpoint(self.mlp, x)
        return x

THE FIX 2 — Flash Attention reduces memory without recomputation:
# FlashAttention uses O(sqrt(N)) memory instead of O(N²)
# For seq_len=32768 saves ~32GB — may remove need for checkpointing
from flash_attn import flash_attn_func  # ROCm compatible

THE FIX 3 — 8-bit optimizer saves memory:
args = TrainingArguments(optim="paged_adamw_8bit")
# Saves ~4x optimizer memory → more room for activations

EXPECTED IMPROVEMENT: Recomputation 41% → 8% | Memory 99.9% → 74% | Speed 1.5x""",
        "bottleneck_type": "gradient_checkpointing_overhead",
        "severity": "high",
    },
]

QA_PAIRS = [
    {
        "q": "What is warp divergence and how does it hurt AMD GPU performance?",
        "a": """Warp divergence occurs when threads in the same warp take different
branches (if/else). AMD MI300X runs 64 threads per warp in lockstep (SIMD).

When divergence happens:
- GPU runs 'if' path with half threads active (other half masked)
- GPU runs 'else' path with other half active (first half masked)
- Result: 2x execution time for same work

DETECT: rocprof shows "Warp Execution Efficiency" < 100%
Values below 80% = significant divergence problem.

FIX: Use branchless ternary operations, sort data so same-warp
threads take same path, use predicated execution."""
    },
    {
        "q": "What is the difference between rocprof and omniperf?",
        "a": """Both are AMD profiling tools but for different levels of analysis:

ROCPROF — Fast, lightweight:
  Use for: finding WHICH kernels are slow
  Command: rocprof --stats python train.py
  Output: kernel timing, basic counters

OMNIPERF — Deep architectural analysis:
  Use for: understanding WHY a kernel is slow
  Command: omniperf profile -n run -- python train.py
           omniperf analyze -p workloads/run/
  Output: memory hierarchy, cache, occupancy details

WORKFLOW:
1. rocprof → identify hot kernels
2. omniperf → deep dive those kernels
3. Fix → re-run rocprof to verify"""
    },
    {
        "q": "How do I profile a PyTorch model on AMD ROCm?",
        "a": """Three methods from simplest to most detailed:

METHOD 1 — PyTorch Profiler (no extra tools needed):
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True) as prof:
    output = model(input)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

METHOD 2 — rocprof (AMD specific):
rocprof --stats python train.py
rocprof --stats -o output.csv python train.py  # save to CSV

METHOD 3 — omniperf (deepest):
omniperf profile -n myrun -- python train.py
omniperf analyze -p workloads/myrun/

START with Method 1 for quick wins, use Method 3 for kernel optimization."""
    },
]


def main():
    print("=" * 60)
    print("ProfiloAI - Extended Training Data Generator")
    print("=" * 60)

    sft_pairs = []

    print(f"\n📊 Processing {len(EXTENDED_SAMPLES)} bottleneck scenarios...")
    for sample in EXTENDED_SAMPLES:
        user_msg = f"Analyze this AMD GPU profiler output:\n\n{sample['profiler_output']}"
        sft_pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": sample["diagnosis"]},
            ],
            "text": f"<s>[INST] {user_msg} [/INST] {sample['diagnosis']} </s>",
            "bottleneck_type": sample["bottleneck_type"],
        })

    print(f"📝 Processing {len(QA_PAIRS)} Q&A pairs...")
    for qa in QA_PAIRS:
        sft_pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": qa["q"]},
                {"role": "assistant", "content": qa["a"]},
            ],
            "text": f"<s>[INST] {qa['q']} [/INST] {qa['a']} </s>",
            "bottleneck_type": "knowledge_qa",
        })

    output_file = OUTPUT_DIR / "extended_samples.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sft_pairs:
            f.write(json.dumps(item) + "\n")

    print(f"\n✅ Saved {len(sft_pairs)} extended samples → {output_file}")
    print("\n📊 Bottleneck types covered:")
    for s in EXTENDED_SAMPLES:
        print(f"  ✅ {s['bottleneck_type']}")
    print("\n🎉 Done! Re-run scripts/03_clean_and_format.py to include this data.")


if __name__ == "__main__":
    main()
