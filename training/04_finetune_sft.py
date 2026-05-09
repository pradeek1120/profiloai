"""LoRA SFT fine-tuning for ProfiloAI on AMD MI300X."""
import inspect
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/sft_dataset.jsonl"
OUTPUT_DIR = ROOT / "outputs/profiloai-sft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_CONFIG = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)


def build_sft_config():
    config_values = {
        "output_dir": str(OUTPUT_DIR),
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "fp16": False,
        "max_seq_length": 2048,
        "max_length": 2048,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": 200,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "dataset_text_field": "text",
        "packing": True,
        "group_by_length": True,
    }
    valid_keys = inspect.signature(SFTConfig).parameters
    filtered = {key: value for key, value in config_values.items() if key in valid_keys}
    return SFTConfig(**filtered)

def train():
    print("ProfiloAI - SFT Fine-Tuning on AMD MI300X")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}. Run scripts 01, 02, 09, and 03 first.")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  No GPU detected. This script is intended for AMD Developer Cloud MI300X.")

    data = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            data.append({"text": json.loads(line)["text"]})
    if len(data) < 2:
        raise ValueError("Need at least 2 SFT examples to create train/eval splits.")

    val_size = max(1, int(len(data) * 0.1))
    train_ds = Dataset.from_list(data[val_size:])
    val_ds = Dataset.from_list(data[:val_size])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.enable_input_require_grads()

    training_config = build_sft_config()
    trainer_kwargs = {
        "model": model,
        "args": training_config,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "peft_config": LORA_CONFIG,
    }
    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"✅ Done! Loss: {result.metrics.get('train_loss','N/A')}")

if __name__ == "__main__":
    train()
