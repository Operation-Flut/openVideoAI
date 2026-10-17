"""
CLI entry point for openVideoAI.
Provides headless batch processing capabilities.
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import create_app
from src.services.model_service import ModelService
from src.services.processing_service import ProcessingService
from src.services.settings_service import SettingsService
from src.models.registry import get_registry
from src.processing.pipeline.registry import get_pipeline_registry
from src.core.paths import exports_dir, models_dir
from src.api.schemas import ProcessRequest, PipelineStep, OutputFormat, TaskStatus

app = typer.Typer(
    name="openvideoai",
    help="Local AI Video Enhancement Toolkit - CLI",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the API server."""
    import uvicorn
    console.print(f"[green]Starting openVideoAI server on {host}:{port}[/green]")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)


@app.command()
def models(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all models"),
    downloaded: bool = typer.Option(False, "--downloaded", "-d", help="List only downloaded models"),
    feature: Optional[str] = typer.Option(None, "--feature", "-f", help="Filter by feature"),
):
    """Manage models."""
    model_service = ModelService()

    if list_all or downloaded or feature:
        models_data = model_service.get_all_models_with_status()

        if downloaded:
            models_data = [m for m in models_data if m.get("downloaded")]

        if feature:
            models_data = [m for m in models_data if feature in m.get("features", [])]

        table = Table(title="Models")
        table.add_column("Name", style="cyan")
        table.add_column("Feature", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Path", style="blue")

        for m in models_data:
            status = "✓ Downloaded" if m.get("downloaded") else "✗ Not downloaded"
            features = ", ".join(m.get("features", []))
            path = m.get("local_path") or "-"
            table.add_row(m["name"], features, m["size"], status, path)

        console.print(table)
    else:
        console.print("[yellow]Use --list, --downloaded, or --feature to list models[/yellow]")


@app.command()
def download(
    model_name: str = typer.Argument(..., help="Model name to download"),
    force: bool = typer.Option(False, "--force", help="Force re-download"),
):
    """Download a model."""
    model_service = ModelService()

    model = model_service.get_model(model_name)
    if not model:
        console.print(f"[red]Model not found: {model_name}[/red]")
        raise typer.Exit(1)

    if model.is_builtin:
        console.print(f"[yellow]{model_name} is built-in, no download needed[/yellow]")
        return

    if model_service.is_downloaded(model_name) and not force:
        console.print(f"[yellow]{model_name} already downloaded. Use --force to re-download.[/yellow]")
        return

    console.print(f"[green]Downloading {model_name}...[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {model_name}", total=100)

        def update_progress(p):
            progress.update(task, completed=p.progress_percent, description=f"{p.status}: {p.progress_percent:.1f}%")

        success = asyncio.run(model_service.download_model(model_name, force, update_progress))

    if success:
        console.print(f"[green]Successfully downloaded {model_name}[/green]")
    else:
        console.print(f"[red]Failed to download {model_name}[/red]")
        raise typer.Exit(1)


@app.command()
def delete_model(
    model_name: str = typer.Argument(..., help="Model name to delete"),
):
    """Delete a downloaded model."""
    model_service = ModelService()

    if not model_service.is_downloaded(model_name):
        console.print(f"[yellow]{model_name} is not downloaded[/yellow]")
        return

    success = model_service.delete_model(model_name)
    if success:
        console.print(f"[green]Deleted {model_name}[/green]")
    else:
        console.print(f"[red]Failed to delete {model_name}[/red]")
        raise typer.Exit(1)


@app.command()
def enhance(
    input_path: Path = typer.Argument(..., help="Input file or directory", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file or directory"),
    upscale_model: Optional[str] = typer.Option(None, "--upscale-model", help="Upscale model to use"),
    upscale_scale: int = typer.Option(4, "--upscale-scale", help="Upscale factor"),
    face_model: Optional[str] = typer.Option(None, "--face-model", help="Face recovery model"),
    face_fidelity: float = typer.Option(0.5, "--face-fidelity", help="Face fidelity weight (0-1)"),
    sharpen_model: Optional[str] = typer.Option(None, "--sharpen-model", help="Sharpen model"),
    sharpen_strength: float = typer.Option(1.0, "--sharpen-strength", help="Sharpen strength"),
    denoise_model: Optional[str] = typer.Option(None, "--denoise-model", help="Denoise model"),
    denoise_strength: float = typer.Option(0.5, "--denoise-strength", help="Denoise strength"),
    interp_model: Optional[str] = typer.Option(None, "--interp-model", help="Frame interpolation model"),
    interp_multiplier: int = typer.Option(2, "--interp-multiplier", help="Frame rate multiplier"),
    color_model: Optional[str] = typer.Option(None, "--color-model", help="Color transfer model"),
    color_reference: Optional[Path] = typer.Option(None, "--color-reference", help="Reference image for color transfer"),
    color_strength: float = typer.Option(1.0, "--color-strength", help="Color transfer strength"),
    lowlight_model: Optional[str] = typer.Option(None, "--lowlight-model", help="Low-light model"),
    lowlight_strength: float = typer.Option(0.7, "--lowlight-strength", help="Low-light strength"),
    output_format: str = typer.Option("source", "--format", help="Output format"),
    quality: str = typer.Option("medium", "--quality", help="Output quality"),
    device: str = typer.Option("auto", "--device", help="Compute device"),
    batch: bool = typer.Option(False, "--batch", help="Process directory as batch"),
):
    """Enhance image or video file(s)."""
    processing_service = ProcessingService()

    # Build pipeline steps
    steps = []

    if upscale_model:
        steps.append(PipelineStep(
            feature="Upscale",
            model=upscale_model,
            parameters={"scale": upscale_scale, "device": device}
        ))

    if face_model:
        steps.append(PipelineStep(
            feature="Face Recovery",
            model=face_model,
            parameters={"fidelity_weight": face_fidelity, "device": device}
        ))

    if sharpen_model:
        steps.append(PipelineStep(
            feature="Sharpen / Deblur",
            model=sharpen_model,
            parameters={"strength": sharpen_strength, "device": device}
        ))

    if denoise_model:
        steps.append(PipelineStep(
            feature="Denoise",
            model=denoise_model,
            parameters={"strength": denoise_strength, "device": device}
        ))

    if interp_model:
        steps.append(PipelineStep(
            feature="Frame Interpolation",
            model=interp_model,
            parameters={"multiplier": interp_multiplier, "device": device}
        ))

    if color_model:
        if not color_reference:
            console.print("[red]Color transfer requires --color-reference[/red]")
            raise typer.Exit(1)
        steps.append(PipelineStep(
            feature="Match Color",
            model=color_model,
            parameters={
                "strength": color_strength,
                "reference_path": str(color_reference),
                "device": device
            }
        ))

    if lowlight_model:
        steps.append(PipelineStep(
            feature="Low-Light Enhancement",
            model=lowlight_model,
            parameters={"strength": lowlight_strength, "device": device}
        ))

    if not steps:
        console.print("[yellow]No enhancement steps specified. Use --help for options.[/yellow]")
        raise typer.Exit(1)

    # Process files
    if batch and input_path.is_dir():
        # Batch process directory
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        all_exts = video_exts | image_exts

        files = [f for f in input_path.iterdir() if f.suffix.lower() in all_exts]
        if not files:
            console.print("[yellow]No supported media files found[/yellow]")
            return

        output_dir = output or (input_path / "enhanced")
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[green]Processing {len(files)} files...[/green]")

        for file in files:
            out_file = output_dir / f"{file.stem}_enhanced{file.suffix}"
            request = ProcessRequest(
                input_path=str(file),
                output_path=str(out_file),
                steps=steps,
                output_format=OutputFormat(output_format),
                quality=quality,
            )

            task = processing_service.create_task(request)
            asyncio.run(processing_service.start_processing(task.task_id, request))

            # Wait for completion
            while True:
                status = processing_service.get_task_status(task.task_id)
                if status.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ABORTED):
                    break
                asyncio.sleep(1)

            if status.status == TaskStatus.COMPLETED:
                console.print(f"[green]✓[/green] {file.name} -> {out_file.name}")
            else:
                console.print(f"[red]✗[/red] {file.name}: {status.error}")

    else:
        # Single file
        out_path = output
        if not out_path:
            export_dir = exports_dir()
            export_dir.mkdir(parents=True, exist_ok=True)
            suffix = input_path.suffix if output_format == "source" else f".{output_format}"
            out_path = export_dir / f"{input_path.stem}_enhanced{suffix}"

        request = ProcessRequest(
            input_path=str(input_path),
            output_path=str(out_path),
            steps=steps,
            output_format=OutputFormat(output_format),
            quality=quality,
        )

        console.print(f"[green]Processing {input_path.name}...[/green]")

        task = processing_service.create_task(request)
        asyncio.run(processing_service.start_processing(task.task_id, request))

        # Wait with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Processing", total=100)

            while True:
                status = processing_service.get_task_status(task.task_id)
                progress.update(task_id, completed=status.progress, description=status.message)
                if status.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ABORTED):
                    break
                asyncio.sleep(0.5)

        if status.status == TaskStatus.COMPLETED:
            console.print(f"[green]Done! Output: {status.output_path}[/green]")
        else:
            console.print(f"[red]Failed: {status.error}[/red]")
            raise typer.Exit(1)


@app.command()
def presets(
    list_presets: bool = typer.Option(False, "--list", "-l", help="List available presets"),
    save: Optional[str] = typer.Option(None, "--save", help="Save current pipeline as preset"),
):
    """Manage processing presets."""
    presets_dir = Path.home() / ".openvideoai" / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)

    if list_presets:
        presets = list(presets_dir.glob("*.json"))
        if not presets:
            console.print("[yellow]No presets found[/yellow]")
            return

        table = Table(title="Presets")
        table.add_column("Name", style="cyan")
        table.add_column("File", style="blue")

        for p in presets:
            table.add_row(p.stem, p.name)

        console.print(table)

    elif save:
        console.print("[yellow]Preset saving not yet implemented[/yellow]")


@app.command()
def info():
    """Show system information."""
    model_service = ModelService()
    status = model_service.get_system_status()

    from src.utils.system_resources import get_system_info
    sys_info = get_system_info()

    console.print(Panel.fit(
        f"[bold]openVideoAI[/bold] System Information\n\n"
        f"Models: {status['downloaded']}/{status['total_models']} downloaded\n"
        f"Models directory: {status['models_dir']}\n"
        f"CPU threads: {sys_info.get('cpu_threads', 'N/A')}\n"
        f"RAM: {sys_info.get('ram_gb', 'N/A'):.1f} GB\n"
        f"GPU: {sys_info.get('gpu', {}).get('name', 'N/A') if sys_info.get('gpu') else 'None'}\n"
        f"VRAM: {sys_info.get('vram_gb', 'N/A'):.1f} GB" if sys_info.get('vram_gb') else "VRAM: N/A",
        title="System Info",
        border_style="green",
    ))


if __name__ == "__main__":
    app()