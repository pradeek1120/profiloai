"""Basic model evaluation for benchmark sanity checks."""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs/profiloai-dpo/final"
RESULTS_DIR = ROOT / "evaluation/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {"id":"T1","input":"Memory Bandwidth: 14.2%\nL2 Cache Hit: 31%\nOccupancy: 48%","must_have":["bandwidth","coalesced","fix"]},
    {"id":"T2","input":"Occupancy: 21.4%\nVGPR: 192\nShared Memory: 100%","must_have":["occupancy","VGPR","fix"]},
    {"id":"T3","input":"GPU Util: 18%\nCPU: 98%\nGPU Idle: 82%","must_have":["DataLoader","workers","prefetch"]},
]

def generate(model, tokenizer, prompt):
    inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)

def score(response, case):
    found = [k for k in case["must_have"] if k.lower() in response.lower()]
    return len(found)/len(case["must_have"])

def benchmark(model, tokenizer, name):
    print(f"\n  {name}")
    results = []
    for case in TEST_CASES:
        r = generate(model, tokenizer, f"Analyze:\n{case['input']}")
        s = score(r, case)
        results.append(s)
        print(f"    {case['id']}: {s:.0%}")
    return results

print("ProfiloAI - Evaluation")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
base_scores = benchmark(base, tok, "Base Model")
del base; torch.cuda.empty_cache()

ft_scores = base_scores
if MODEL_PATH.exists():
    ft_tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    ft_model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype=torch.bfloat16, device_map="auto")
    ft_scores = benchmark(ft_model, ft_tok, "ProfiloAI")
    del ft_model; torch.cuda.empty_cache()

base_avg = sum(base_scores)/len(base_scores)
ft_avg = sum(ft_scores)/len(ft_scores)
print(f"\n✅ Base: {base_avg:.1%} | ProfiloAI: {ft_avg:.1%} | Improvement: +{(ft_avg-base_avg)/base_avg*100:.1f}%")
