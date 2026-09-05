"""Generate SFT and DPO pairs from collected profiler samples."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/profiler_samples.jsonl"
OUTPUT = ROOT / "data/raw"
OUTPUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "You are ProfiloAI, an AMD GPU performance expert. Give specific diagnosis with exact fix."
BAD = ["try optimizing your code", "check documentation", "performance seems low"]
random.seed(7)

data = []
if INPUT.exists():
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
else:
    raise FileNotFoundError(f"Missing input dataset: {INPUT}. Run scripts/01_collect_rocprof_data.py first.")

sft_pairs, dpo_pairs = [], []
for s in data:
    user = f"Analyze this AMD GPU profiler output:\n\n{s['profiler_output']}"
    sft_pairs.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": s["diagnosis"]},
            ],
            "text": f"<s>[INST] {user} [/INST] {s['diagnosis']} </s>",
            "bottleneck_type": s.get("bottleneck_type", "unknown"),
        }
    )
    dpo_pairs.append(
        {
            "prompt": f"<s>[INST] {user} [/INST]",
            "chosen": s["diagnosis"] + " </s>",
            "rejected": random.choice(BAD) + " </s>",
        }
    )

with open(OUTPUT / "diagnosis_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in sft_pairs:
        f.write(json.dumps(p) + "\n")
with open(OUTPUT / "dpo_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in dpo_pairs:
        f.write(json.dumps(p) + "\n")

print(f"Created {len(sft_pairs)} SFT pairs | {len(dpo_pairs)} DPO pairs")
