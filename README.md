# ProfiloAI - AMD GPU Performance Doctor

ProfiloAI helps AMD ROCm developers turn profiler output into a clear performance diagnosis. Paste `rocprof`, `omniperf`, or training-loop metrics and the app returns:

- the likely bottleneck
- the root cause in plain English
- a concrete code fix
- an expected speedup range

This project is prepared for the **AMD Developer Hackathon** under **Track 2: Fine-Tuning on AMD GPUs**. The intended final run is on **AMD Developer Cloud with AMD Instinct MI300X**, using ROCm, PyTorch, Hugging Face PEFT/TRL, and vLLM.

## Why It Matters

GPU profilers are powerful, but their output is hard to act on quickly. ProfiloAI acts like a performance assistant for AMD Instinct/ROCm users, especially developers working on PyTorch, Hugging Face, and vLLM workloads.

## Hackathon Fit

ProfiloAI matches the Fine-Tuning on AMD GPUs track because it is a domain-specific LLM assistant for GPU performance diagnosis. The final submission should show:

- AMD Developer Cloud usage
- MI300X / ROCm environment evidence
- LoRA SFT training on AMD GPU
- DPO alignment or preference tuning
- vLLM serving
- a working Gradio demo URL
- benchmark results generated after training

Use [AMD_CLOUD_RUNBOOK.md](AMD_CLOUD_RUNBOOK.md) as the step-by-step cloud checklist.

## Demo

The Gradio app works in two modes:

1. **Model mode:** calls a local vLLM OpenAI-compatible server at `localhost:8000`.
2. **Fallback demo mode:** if vLLM is not running, it still gives deterministic diagnoses for common AMD GPU bottlenecks. This keeps the hosted demo usable for judges, but the final submission should still include AMD Cloud training evidence.

Run the demo locally:

```bash
cd profiloai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open the printed Gradio URL, load an example, and click **Diagnose**.

## Full AMD Cloud / MI300X Flow

Use this path on AMD Developer Cloud to train, benchmark, and collect submission evidence.

```bash
cd profiloai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
cp .env.example .env
```

Fill in `.env`, especially `HF_TOKEN`.

Check and save AMD Cloud evidence:

```bash
python3 scripts/00_check_amd_cloud.py
```

Generate datasets:

```bash
python3 scripts/01_collect_rocprof_data.py
python3 scripts/09_more_training_data.py
python3 scripts/02_generate_diagnosis_pairs.py
python3 scripts/03_clean_and_format.py
```

Run local sanity checks:

```bash
python3 tests/local_test.py
```

Fine-tune and align:

```bash
python3 training/04_finetune_sft.py
python3 training/05_dpo_alignment.py
```

Evaluate:

```bash
python3 evaluation/06_evaluate.py
python3 evaluation/benchmark_comparison.py
```

Serve with vLLM and launch the UI:

```bash
python3 serving/07_serve_vllm.py
python3 ui/08_gradio_ui.py
```

## Project Structure

```text
profiloai/
├── app.py                         # Hugging Face Spaces / Gradio entrypoint
├── AMD_CLOUD_RUNBOOK.md           # Exact AMD Cloud run and evidence checklist
├── README.md
├── SUBMISSION.md                  # Copy-ready hackathon submission notes
├── requirements.txt               # Demo dependencies
├── requirements-training.txt      # MI300X training/serving dependencies
├── scripts/
│   ├── 00_check_amd_cloud.py       # Saves ROCm/MI300X readiness evidence
│   ├── 01_collect_rocprof_data.py
│   ├── 02_generate_diagnosis_pairs.py
│   ├── 03_clean_and_format.py
│   └── 09_more_training_data.py
├── training/
│   ├── 04_finetune_sft.py
│   └── 05_dpo_alignment.py
├── evaluation/
│   ├── 06_evaluate.py
│   └── benchmark_comparison.py
├── serving/
│   └── 07_serve_vllm.py
├── ui/
│   └── 08_gradio_ui.py
├── tests/
│   └── local_test.py
├── data/
│   ├── raw/
│   └── processed/
└── outputs/                       # ignored; model checkpoints go here
```

## Bottlenecks Covered

- low memory bandwidth
- low GPU occupancy
- high VGPR / register pressure
- DataLoader CPU bottleneck
- FP32 instead of BF16
- warp divergence
- PCIe transfer overhead
- kernel launch overhead
- atomic collisions
- multi-GPU all-reduce overhead
- matrix dimension misalignment
- cache thrashing
- gradient checkpointing overhead

## Hackathon Submission Checklist

- Public GitHub repository
- Working Gradio or Hugging Face Space URL
- AMD Developer Cloud / MI300X evidence
- ROCm + PyTorch GPU detection evidence
- Training logs from `training/04_finetune_sft.py`
- DPO logs from `training/05_dpo_alignment.py`
- Benchmark report from `evaluation/benchmark_comparison.py`
- Video presentation showing the demo
- Slide presentation explaining problem, solution, AMD stack, and business value
- Cover image
- Short and long descriptions
- Technology/category tags
- MIT license

Useful copy for the submission is in [SUBMISSION.md](SUBMISSION.md).

## Current Status

This repo is a clean working prototype and is ready to run on AMD Developer Cloud. The demo is usable without GPU through fallback mode, but final performance claims should come from the MI300X training and benchmark run described in [AMD_CLOUD_RUNBOOK.md](AMD_CLOUD_RUNBOOK.md).

## One-Line Pitch

ProfiloAI turns AMD GPU profiler output into one actionable performance fix in seconds.
