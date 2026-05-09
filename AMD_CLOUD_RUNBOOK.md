# AMD Developer Cloud Runbook

Use this file when running ProfiloAI on AMD Developer Cloud with MI300X credits. It gives you the exact evidence and commands needed for the AMD Developer Hackathon submission.

## Goal

Show that ProfiloAI uses AMD Developer Cloud, ROCm, PyTorch, Hugging Face fine-tuning, and vLLM on MI300X.

## 1. Start The AMD Cloud Instance

Recommended setup:

- AMD Instinct MI300X instance
- ROCm/PyTorch image if available
- Enough disk space for the base model and LoRA checkpoints
- SSH access enabled

After connecting:

```bash
git clone <your-public-github-repo-url>
cd profiloai
```

If you upload files manually, make sure you are inside the `profiloai` folder before running commands.

## 2. Install Training Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-training.txt
```

If the cloud image already includes ROCm PyTorch, keep that version and only install the missing packages:

```bash
pip install transformers datasets accelerate peft trl vllm wandb gradio requests python-dotenv matplotlib
```

## 3. Add Secrets

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
HF_TOKEN=your_huggingface_token
BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

Do not commit `.env`.

## 4. Capture AMD Cloud Evidence

Run:

```bash
python3 scripts/00_check_amd_cloud.py
```

Save these for the submission:

- terminal screenshot showing MI300X / ROCm / PyTorch
- `evaluation/results/amd_cloud_check.json`
- screenshot of AMD Developer Cloud instance page if allowed

## 5. Generate Datasets

```bash
python3 scripts/01_collect_rocprof_data.py
python3 scripts/09_more_training_data.py
python3 scripts/02_generate_diagnosis_pairs.py
python3 scripts/03_clean_and_format.py
python3 tests/local_test.py
```

Expected local check result:

```text
Results: 14/14 passed | 0 failed
```

## 6. Fine-Tune On MI300X

```bash
python3 training/04_finetune_sft.py
```

Expected output folder:

```text
outputs/profiloai-sft/final
```

Then run DPO alignment:

```bash
python3 training/05_dpo_alignment.py
```

Expected output folder:

```text
outputs/profiloai-dpo/final
```

## 7. Benchmark

```bash
python3 evaluation/06_evaluate.py
python3 evaluation/benchmark_comparison.py
```

Save generated files from:

```text
evaluation/results/
```

Use the benchmark report in your slides and video only after it is generated on MI300X.

## 8. Serve With vLLM

Terminal 1:

```bash
python3 serving/07_serve_vllm.py
```

Terminal 2:

```bash
python3 ui/08_gradio_ui.py
```

Open the Gradio URL and test the examples.

## 9. Submission Evidence Checklist

Include or mention:

- Public GitHub repository
- Working demo URL
- AMD Cloud / MI300X evidence
- ROCm/PyTorch evidence
- Training logs
- Benchmark report
- Demo video
- Slides
- MIT license

## 10. Final Claim To Use

Use this only after running the cloud steps:

```text
ProfiloAI was fine-tuned and benchmarked on AMD Developer Cloud using AMD Instinct MI300X, ROCm, PyTorch, Hugging Face PEFT/TRL, and served through vLLM with a Gradio interface.
```
