"""
Denoise enhancement pipelines.
Supports SCUNet, NAFNet-SIDD, NAFNet-REDS, NAFNet-GoPro, MPRNet.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, ModelLoadError, InferenceError, ProcessingAborted
from ...processing.pipeline.tiling import process_tiled
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)


class DenoisePipeline(EnhancementPipeline):
    """Denoise enhancement pipeline."""

    config = PipelineConfig(
        name="DenoisePipeline",
        feature="Denoise",
        description="Remove noise and compression artifacts from images/videos",
        supported_formats=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".webm"],
        supports_video=True,
        supports_image=True,
        requires_gpu=False,  # Some models run on CPU
        default_parameters={
            "strength": 0.5,
            "tile_size": 256,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "tile_size": {"type": "integer", "minimum": 64, "maximum": 1024, "default": 256},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        strength: float = 0.5,
        tile_size: int = 256,
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Denoise")
        super().__init__(model_name, device, parameters=parameters, task_id=task_id)
        self.strength = max(0.0, min(1.0, strength))
        self.tile_size = max(64, int(tile_size))
        self._model = None
        # Architecture and tunables come from the models.json catalog entry.
        self._model_config = get_registry().runtime_config(model_name, "Denoise")

    def _effective_params(self) -> Dict[str, Any]:
        """Step parameters merged with the resolved constructor values."""
        return {
            "strength": self.strength,
            "tile_size": self.tile_size,
            **self.parameters,
        }

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load denoise model."""
        try:
            import torch

            if isinstance(model_path, dict):
                main_path = model_path.get("main")
            else:
                main_path = model_path

            if not main_path:
                raise ModelLoadError(f"No model file found for {self.model_name}")

            device_obj = torch.device(self.get_device())
            arch = self._model_config['arch']

            if arch == 'SCUNet':
                from ..archs.scunet_arch import SCUNet
                self._model = SCUNet()
            elif arch == 'NAFNet':
                from ..archs.nafnet_arch import from_runtime_config
                self._model = from_runtime_config(self._model_config)
            elif arch == 'MPRNet':
                from ..archs.mprnet_arch import MPRNet
                self._model = MPRNet()
            else:
                raise ModelLoadError(f"Unknown architecture: {arch}")

            state_dict = torch.load(str(main_path), map_location='cpu')
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']

            self._model.load_state_dict(state_dict, strict=True)
            self._model.eval()
            self._model.to(device_obj)

            self._loaded = True
            logger.info(f"Loaded {self.model_name} ({arch}) on {self.get_device()}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process image or video for denoising."""
        if not self._model:
            raise InferenceError("Model not loaded. Call load_model() first.")

        path = context.input_path
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

        if is_video:
            return self._process_video(context)
        else:
            return self._process_image(context)

    def _run_model(self, tile: np.ndarray) -> np.ndarray:
        """Run the model on an RGB uint8 tile and return RGB uint8 output."""
        import torch

        tile_tensor = (
            torch.from_numpy(np.ascontiguousarray(tile))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
        )
        if hasattr(self._model, 'parameters'):
            tile_tensor = tile_tensor.to(next(self._model.parameters()).device)
        else:
            tile_tensor = tile_tensor.to(torch.device(self.get_device()))

        with torch.no_grad():
            out = self._model(tile_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]

        base = tile_tensor.squeeze(0).permute(1, 2, 0).float().cpu()
        out = out.squeeze(0).permute(1, 2, 0).float().cpu()
        if self.strength < 1.0:
            out = base * (1 - self.strength) + out * self.strength
        return out.mul(255.0).clamp(0, 255).round().numpy().astype(np.uint8)

    def _process_image_tiled(self, img: np.ndarray, context: ProcessingContext) -> np.ndarray:
        """Process a large image in overlapping tiles with windowed blending."""
        def run_tile(tile: np.ndarray) -> np.ndarray:
            if context is not None and context.abort_flag:
                raise ProcessingAborted("Image processing aborted")
            return self._run_model(tile)

        return process_tiled(
            img,
            run_tile,
            tile_size=self.tile_size,
            overlap=max(32, min(128, max(64, self.tile_size // 2))),
            window='hamming',
            progress=context.report_progress if context is not None else None,
        )

    def _process_image(self, context: ProcessingContext) -> ProcessingResult:
        """Process a single image, tiling it when it exceeds the tile size."""
        try:
            img = cv2.imread(str(context.input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {context.input_path}")

            context.report_progress(10, "Preprocessing")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            context.report_progress(30, "Running denoising")
            if max(img_rgb.shape[:2]) > self.tile_size:
                output = self._process_image_tiled(img_rgb, context)
            else:
                output = self._run_model(img_rgb)

            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            context.report_progress(80, "Saving output")

            write_params = []
            if context.output_path.suffix.lower() in {'.jpg', '.jpeg'}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

            if not cv2.imwrite(str(context.output_path), output_bgr, write_params):
                raise InferenceError(f"Failed to write output: {context.output_path}")

            context.report_progress(100, "Denoising complete")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    **self._effective_params(),
                    'model_arch': self._model_config['arch'],
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Denoising failed: {e}")

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        """Process video for denoising."""
        import torch

        try:
            cap = cv2.VideoCapture(str(context.input_path))
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {context.input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(context.output_path), fourcc, fps, (width, height))

            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {context.output_path}")

            processed = 0
            device = next(self._model.parameters()).device

            context.report_progress(5, f"Processing {total_frames} frames")

            while True:
                if context.abort_flag:
                    raise ProcessingAborted("Video processing aborted")

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

                    with torch.no_grad():
                        if self._model_config['arch'] == 'MPRNet':
                            output = self._model(frame_tensor)[0]
                        else:
                            output = self._model(frame_tensor)

                    if self.strength < 1.0:
                        output = frame_tensor * (1 - self.strength) + output * self.strength

                    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    output = (output * 255).clip(0, 255).astype(np.uint8)
                    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                    out.write(output_bgr)
                    processed += 1

                except Exception as e:
                    logger.warning(f"Frame {processed} denoising failed: {e}")
                    out.write(frame)

                if processed % 30 == 0 or processed == total_frames:
                    progress = 5 + int((processed / total_frames) * 95)
                    context.report_progress(progress, f"Frame {processed}/{total_frames}")

            cap.release()
            out.release()

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'frames_processed': processed,
                    'total_frames': total_frames,
                    'strength': self.strength,
                    'fps': fps,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Video denoising failed: {e}")


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Denoise",
    pipeline_class=DenoisePipeline,
    available_models=get_registry().get_available_model_names("Denoise"),
    default_model=get_registry().get_default_model("Denoise"),
)