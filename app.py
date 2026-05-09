"""Hugging Face Spaces entrypoint for the ProfiloAI Gradio demo."""
from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).parent / "ui/08_gradio_ui.py"), run_name="__main__")
