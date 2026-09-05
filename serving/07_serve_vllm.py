"""Start the vLLM OpenAI-compatible server for ProfiloAI."""
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "outputs/profiloai-dpo/final"
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
PORT = 8000

adapter_config = ADAPTER_PATH / "adapter_config.json"
use_lora_adapter = adapter_config.exists()

if use_lora_adapter:
    print(f"Starting vLLM server with base model: {BASE_MODEL}")
    print(f"Loading ProfiloAI LoRA adapter: {ADAPTER_PATH}")
else:
    print(f"Starting vLLM server with: {BASE_MODEL}")

cmd = [
    sys.executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    BASE_MODEL,
    "--served-model-name",
    "profiloai",
    "--dtype",
    "bfloat16",
    "--port",
    str(PORT),
    "--host",
    "0.0.0.0",
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.90",
    "--trust-remote-code",
]

if use_lora_adapter:
    cmd.extend([
        "--enable-lora",
        "--lora-modules",
        f"profiloai={ADAPTER_PATH}",
    ])

try:
    process = subprocess.Popen(cmd)
    print(f"✅ Server starting at http://localhost:{PORT}")
    print("Run: python ui/08_gradio_ui.py in a new terminal")
    process.wait()
except KeyboardInterrupt:
    process.terminate()
    print("Stopped.")
