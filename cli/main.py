#!/usr/bin/env python3
"""
CLI for openVideoAI - Headless batch processing
Usage:
  openvideoai enhance input.mp4 --upscale 4 --face-recovery --output out.mp4
  openvideoai batch ./input/ --pipeline preset_cinematic --output ./enhanced/
  openvideoai preset save "my_4k" --upscale RealESRGAN_x4plus --scale 4
"""
import typer
from pathlib import Path

app = typer.Typer(help="openVideoAI - Local AI Video Enhancement CLI")

@app.command()
def enhance(
    input_path: Path,
    output_path: Path = None,
    upscale: int = 4,
    face_recovery: bool = False,
    sharpen: bool = False,
    denoise: bool = False,
):
    """Enhance a single file."""
    typer.echo(f"Enhancing {input_path} -> {output_path or 'auto'}")

@app.command()
def batch(
    input_dir: Path,
    output_dir: Path = Path("./enhanced"),
    preset: str = "default",
):
    """Batch process a directory."""
    typer.echo(f"Batch processing {input_dir} with preset '{preset}'")

if __name__ == "__main__":
    app()
