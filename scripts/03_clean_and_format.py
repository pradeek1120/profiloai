"""Clean, validate, and format final JSONL datasets."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw"
OUTPUT = ROOT / "data/processed"
OUTPUT.mkdir(parents=True, exist_ok=True)
random.seed(7)


def load(path):
    data = []
    if not Path(path).exists():
        return data
    with open(path, encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return data


sft = load(INPUT / "diagnosis_pairs.jsonl") + load(INPUT / "extended_samples.jsonl")
dpo = load(INPUT / "dpo_pairs.jsonl")

seen, unique_sft = set(), []
for item in sft:
    if "text" not in item or "messages" not in item:
        raise ValueError(f"SFT item missing required fields: {item}")
    key = item.get("text", "")[:160]
    if key not in seen:
        seen.add(key)
        unique_sft.append(item)

for item in dpo:
    missing = {"prompt", "chosen", "rejected"} - set(item)
    if missing:
        raise ValueError(f"DPO item missing required fields {missing}: {item}")

if not unique_sft:
    raise ValueError("No SFT data found. Run scripts 01, 02, and 09 first.")
if not dpo:
    raise ValueError("No DPO data found. Run scripts 01 and 02 first.")

random.shuffle(unique_sft)
random.shuffle(dpo)

with open(OUTPUT / "sft_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in unique_sft:
        f.write(json.dumps(item) + "\n")
with open(OUTPUT / "dpo_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dpo:
        f.write(json.dumps(item) + "\n")

summary = {
    "sft_examples": len(unique_sft),
    "dpo_examples": len(dpo),
    "bottleneck_types": sorted({item.get("bottleneck_type", "unknown") for item in unique_sft}),
}
with open(OUTPUT / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"SFT: {len(unique_sft)} | DPO: {len(dpo)} | Summary: {OUTPUT / 'dataset_summary.json'}")
