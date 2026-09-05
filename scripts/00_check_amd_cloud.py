"""Check AMD Developer Cloud / ROCm readiness and save non-secret evidence.

Run this first on the MI300X instance. It prints a readable summary and writes
``evaluation/results/amd_cloud_check.json`` for your submission evidence.
"""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_command(command):
    exe = shutil.which(command[0])
    if not exe:
        return {"available": False, "command": command, "output": ""}

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        return {
            "available": True,
            "command": command,
            "returncode": result.returncode,
            "output": output[:4000],
        }
    except Exception as exc:
        return {
            "available": True,
            "command": command,
            "error": f"{type(exc).__name__}: {exc}",
        }


def torch_info():
    try:
        import torch
    except ImportError:
        return {"installed": False}

    info = {
        "installed": True,
        "version": getattr(torch, "__version__", "unknown"),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available_api": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return info


def main():
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch_info(),
        "commands": {
            "rocm_smi": run_command(["rocm-smi", "--showproductname"]),
            "rocminfo": run_command(["rocminfo"]),
            "hipcc": run_command(["hipcc", "--version"]),
        },
    }

    output_file = RESULTS_DIR / "amd_cloud_check.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("ProfiloAI AMD Cloud Check")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    torch_report = report["torch"]
    if torch_report["installed"]:
        print(f"PyTorch: {torch_report['version']}")
        print(f"ROCm/HIP: {torch_report['hip_version']}")
        print(f"GPU available through torch: {torch_report['cuda_available_api']}")
        for device in torch_report["devices"]:
            print(f"GPU {device['index']}: {device['name']} ({device['total_memory_gb']} GB)")
    else:
        print("PyTorch: not installed")

    for name, command_report in report["commands"].items():
        state = "found" if command_report.get("available") else "not found"
        print(f"{name}: {state}")

    print(f"\nSaved evidence: {output_file}")


if __name__ == "__main__":
    main()
