"""DPO alignment after SFT fine-tuning."""
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "outputs/profiloai-sft/final"
DPO_DATA = ROOT / "data/processed/dpo_dataset.jsonl"
OUTPUT_DIR = ROOT / "outputs/profiloai-dpo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPO_CONFIG = DPOConfig(output_dir=str(OUTPUT_DIR), beta=0.1, loss_type="sigmoid", num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=5e-7, bf16=True, max_length=2048, max_prompt_length=1024, logging_steps=5, save_strategy="steps", save_steps=100, save_total_limit=2, remove_unused_columns=False)

def train_dpo():
    if not SFT_PATH.exists():
        raise FileNotFoundError(f"Run script 04 first! {SFT_PATH} not found.")
    if not DPO_DATA.exists():
        raise FileNotFoundError(f"Missing DPO dataset: {DPO_DATA}. Run scripts 01, 02, and 03 first.")

    data = []
    with open(DPO_DATA, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    if len(data) < 2:
        raise ValueError("Need at least 2 DPO examples to create train/eval splits.")

    val_size = max(1, int(len(data) * 0.1))
    train_ds = Dataset.from_list(data[val_size:])
    val_ds = Dataset.from_list(data[:val_size])

    tokenizer = AutoTokenizer.from_pretrained(str(SFT_PATH))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")
    ref_base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(SFT_PATH), is_trainable=True)
    ref_model = PeftModel.from_pretrained(ref_base_model, str(SFT_PATH), is_trainable=False)
    model.print_trainable_parameters()

    trainer = DPOTrainer(model=model, ref_model=ref_model, args=DPO_CONFIG, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"✅ DPO done! Model saved to {OUTPUT_DIR / 'final'}")

if __name__ == "__main__":
    train_dpo()
