"""
Upscale enhancement pipelines.
Supports Real-ESRGAN, RealBasicVSR, Real-CUGAN, and other upscale models.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, ModelLoadError, InferenceError, ProcessingAborted
from ..pipeline.video_writer import open_video_sink
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)


class UpscalePipeline(EnhancementPipeline):
    """Upscale enhancement pipeline using Real-ESRGAN and other models."""

    config = PipelineConfig(
        name="UpscalePipeline",
        feature="Upscale",
        description="Increase image/video resolution using AI upscaling models",
        supported_formats=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".webm"],
        supports_video=True,
        supports_image=True,
        requires_gpu=True,
        default_parameters={
            "scale": 4,
            "tile_size": 400,
            "fp16": True,
            "jpg_quality": 95,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "scale": {"type": "integer", "enum": [1, 2, 3, 4, 8], "default": 4},
                "tile_size": {"type": "integer", "minimum": 64, "maximum": 1024, "default": 400},
                "fp16": {"type": "boolean", "default": True},
                "jpg_quality": {"type": "integer", "minimum": 1, "maximum": 100, "default": 95},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        scale: int = 4,
        tile_size: int = 400,
        fp16: bool = True,
        jpg_quality: int = 95,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Upscale")
        super().__init__(model_name, device, task_id=task_id)
        self.target_scale = scale
        self.tile_size = tile_size
        self.fp16 = fp16
        self.jpg_quality = max(1, min(100, jpg_quality))
        self._upsampler = None
        # Architecture and loader hyperparameters come from models.json.
        self._model_config = get_registry().runtime_config(model_name, "Upscale")

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load upscale model."""
        try:
            import torch
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            config = self._model_config

            # Handle both single path and dict of paths
            if isinstance(model_path, dict):
                main_path = model_path.get("main")
            else:
                main_path = model_path

            if not main_path:
                raise ModelLoadError(f"No model file found for {self.model_name}")

            # Create model architecture
            model = RRDBNet(**config['model_params'])

            # Determine device
            device_obj = torch.device(self.get_device())

            # Create upsampler
            self._upsampler = RealESRGANer(
                scale=config['scale'],
                model_path=str(main_path),
                model=model,
                tile=self.tile_size,
                tile_pad=config['tile_pad'],
                pre_pad=config['pre_pad'],
                half=config['half_precision'] and self.fp16 and self.get_device().startswith(('cuda', 'hip', 'xpu', 'dml', 'privateuseone')),
                device=device_obj
            )

            self._loaded = True
            logger.info(f"Loaded {self.model_name} on {self.get_device()}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process image or video."""
        if not self._upsampler:
            raise InferenceError("Model not loaded. Call load_model() first.")

        path = context.input_path
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

        if is_video:
            return self._process_video(context)
        else:
            return self._process_image(context)

    def _process_image(self, context: ProcessingContext) -> ProcessingResult:
        """Process single image with optional progress callback for tiling."""
        import torch

        try:
            # Read image
            img = cv2.imread(str(context.input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {context.input_path}")

            h, w = img.shape[:2]
            max_dim = max(h, w)

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            context.report_progress(10, "Starting upscale")

            # Check if image is large enough to need tiling
            if max_dim > self.tile_size:
                output = self._process_image_tiled(img, context)
            else:
                with torch.no_grad():
                    output, _ = self._upsampler.enhance(img, outscale=self.target_scale)
                context.report_progress(90, "Upscale complete")

            # Convert back to BGR for saving
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            # Save
            write_params = []
            if context.output_path.suffix.lower() in {'.jpg', '.jpeg'}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
            elif context.output_path.suffix.lower() == '.png':
                write_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            elif context.output_path.suffix.lower() == '.webp':
                write_params = [cv2.IMWRITE_WEBP_QUALITY, self.jpg_quality]

            if not cv2.imwrite(str(context.output_path), output, write_params):
                raise InferenceError(f"Failed to write output image: {context.output_path}")

            context.report_progress(100, "Saved output")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'input_shape': img.shape[:2],
                    'output_shape': output.shape[:2],
                    'scale_factor': self.target_scale,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Image upscale failed: {e}")

    def _process_image_tiled(self, img: np.ndarray, context: ProcessingContext) -> np.ndarray:
        """Process large image using custom tiling with progress reporting."""
        import torch

        h, w = img.shape[:2]
        scale = self.target_scale
        tile_size = self.tile_size
        tile_pad = self._model_config.get('tile_pad', 10)

        tiles_x = (w + tile_size - 1) // tile_size
        tiles_y = (h + tile_size - 1) // tile_size
        total_tiles = tiles_x * tiles_y

        out_h, out_w = h * scale, w * scale
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        tile_count = 0

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x_start = tx * tile_size
                y_start = ty * tile_size
                x_end = min(x_start + tile_size, w)
                y_end = min(y_start + tile_size, h)

                x_pad_start = max(0, x_start - tile_pad)
                y_pad_start = max(0, y_start - tile_pad)
                x_pad_end = min(w, x_end + tile_pad)
                y_pad_end = min(h, y_end + tile_pad)

                tile = img[y_pad_start:y_pad_end, x_pad_start:x_pad_end]

                with torch.no_grad():
                    try:
                        upscaled_tile, _ = self._upsampler.enhance(tile, outscale=scale)
                    except Exception as e:
                        if "output_tile" in str(e) or "UnboundLocalError" in str(type(e)):
                            logger.warning(f"Tile processing error, skipping tile at ({tx},{ty}): {e}")
                            continue
                        elif "could not allocate tensor" in str(e).lower():
                            logger.warning(f"Tile VRAM error, skipping tile at ({tx},{ty}): {e}")
                            continue
                        else:
                            raise

                    out_x_start = (x_start - x_pad_start) * scale
                    out_y_start = (y_start - y_pad_start) * scale
                    out_x_end = out_x_start + (x_end - x_start) * scale
                    out_y_end = out_y_start + (y_end - y_start) * scale

                    output[
                        y_start * scale:y_end * scale,
                        x_start * scale:x_end * scale,
                    ] = upscaled_tile[
                        out_y_start:out_y_end,
                        out_x_start:out_x_end,
                    ]

                tile_count += 1
                if context.progress_callback:
                    progress = 10 + int((tile_count / total_tiles) * 80)
                    context.report_progress(progress, f"Processing tile {tile_count}/{total_tiles}")

        return output

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        """Process video frame by frame."""
        import torch

        try:
            cap = cv2.VideoCapture(str(context.input_path))
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {context.input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_width = width * self.target_scale
            out_height = height * self.target_scale

            # The final stage is encoded with the Export dialog's codec,
            # quality, audio and LUT; intermediate stages stay on the fast
            # built-in writer (see pipeline/video_writer.py).
            sink = open_video_sink(context.output_path, fps=fps, size=(out_width, out_height), context=context)

            processed = 0
            failed = 0

            context.report_progress(5, f"Processing {total_frames} frames")

            while True:
                if context.abort_flag:
                    raise ProcessingAborted("Video processing aborted")

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    with torch.no_grad():
                        upscaled, _ = self._upsampler.enhance(frame_rgb, outscale=self.target_scale)

                    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                    sink.write(upscaled_bgr)
                    processed += 1

                except Exception as e:
                    logger.warning(f"Frame {processed} failed: {e}")
                    failed += 1
                    # Fallback: resize with cubic interpolation
                    fallback = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
                    sink.write(fallback)

                if processed % 30 == 0 or processed == total_frames:
                    progress = 5 + int((processed / total_frames) * 95)
                    context.report_progress(progress, f"Frame {processed}/{total_frames}")

            cap.release()
            sink.close()

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'frames_processed': processed,
                    'frames_failed': failed,
                    'total_frames': total_frames,
                    'input_resolution': f"{width}x{height}",
                    'output_resolution': f"{out_width}x{out_height}",
                    'scale_factor': self.target_scale,
                    'fps': fps,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Video upscale failed: {e}")


# Register this pipeline
from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Upscale",
    pipeline_class=UpscalePipeline,
    available_models=get_registry().get_available_model_names("Upscale"),
    default_model=get_registry().get_default_model("Upscale"),
)