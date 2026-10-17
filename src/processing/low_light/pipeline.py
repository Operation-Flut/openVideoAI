"""
Low-light enhancement pipeline.
Model list, architectures and defaults come from the models.json catalog.
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


class LowLightPipeline(EnhancementPipeline):
    """Low-light enhancement pipeline."""

    config = PipelineConfig(
        name="LowLightPipeline",
        feature="Low-Light Enhancement",
        description="Enhance underexposed or night images/videos",
        supported_formats=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".webm"],
        supports_video=True,
        supports_image=True,
        requires_gpu=False,
        default_parameters={
            "strength": 0.7,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        strength: float = 0.7,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Low-Light Enhancement")
        super().__init__(model_name, device, task_id=task_id)
        self.strength = max(0.0, min(1.0, strength))
        self._model = None
        # Architecture comes from the models.json catalog entry; non-AI catalog
        # entries are the classical (no-download) implementations.
        self._model_config = get_registry().runtime_config(model_name, "Low-Light Enhancement")
        self._is_classical = bool(self._model_config.get('is_classical'))

    def _enhance_classical(self, img: np.ndarray) -> np.ndarray:
        # CLAHE + gamma correction + mild contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        # gamma
        inv_gamma = 1.0 / 0.8
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, table)
        # Blend with the original according to strength so 0 = no-op, 1 = full effect
        if self.strength < 1.0:
            enhanced = cv2.addWeighted(img, 1.0 - self.strength, enhanced, self.strength, 0.0)
        return enhanced.astype(np.uint8)

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load low-light enhancement model."""
        try:
            if self._is_classical:
                self._loaded = True
                logger.info(f"Initialized classical low-light enhancement: {self.model_name}")
                return True

            import torch

            if isinstance(model_path, dict):
                main_path = model_path.get("main")
            else:
                main_path = model_path

            if not main_path:
                raise ModelLoadError(f"No model file found for {self.model_name}")

            device_obj = torch.device(self.get_device())
            arch = self._model_config['arch']

            if arch == 'ZeroDCE':
                from ..archs.zerodce_arch import ZeroDCE
                self._model = ZeroDCE()
            else:
                raise ModelLoadError(f"Unknown architecture: {arch}")

            state_dict = torch.load(str(main_path), map_location='cpu')
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']

            try:
                self._model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                logger.warning(f"Strict state_dict load failed for {self.model_name}, attempting non-strict: {e}")
                self._model.load_state_dict(state_dict, strict=False)

            self._model.eval()
            self._model.to(device_obj)

            self._loaded = True
            logger.info(f"Loaded {self.model_name} ({arch}) on {self.get_device()}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, falling back to classical: {e}")
            self._is_classical = True
            self._model = None
            self._loaded = True
            return True

    def process(self, context: ProcessingContext) -> ProcessingResult:
        if not self._loaded:
            raise InferenceError("Model not loaded. Call load_model() first.")

        path = context.input_path
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

        if is_video:
            return self._process_video(context)
        else:
            return self._process_image(context)

    def _process_image(self, context: ProcessingContext) -> ProcessingResult:
        import torch

        try:
            img = cv2.imread(str(context.input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {context.input_path}")

            context.report_progress(10, "Enhancing low-light image")

            if self._is_classical:
                output_bgr = self._enhance_classical(img)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
                device = next(self._model.parameters()).device
                img_tensor = img_tensor.to(device)

                with torch.no_grad():
                    output = self._model(img_tensor)
                    if isinstance(output, tuple):
                        output = output[0]

                if self.strength < 1.0:
                    output = img_tensor * (1 - self.strength) + output * self.strength

                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output = (output * 255).clip(0, 255).astype(np.uint8)
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            write_params = []
            if context.output_path.suffix.lower() in {'.jpg', '.jpeg'}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

            if not cv2.imwrite(str(context.output_path), output_bgr, write_params):
                raise InferenceError(f"Failed to write output: {context.output_path}")

            context.report_progress(100, "Low-light enhancement complete")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={'strength': self.strength, 'model_type': 'classical' if self._is_classical else 'ai'},
                model_used=self.model_name,
                device_used=self.get_device(),
            )
        except Exception as e:
            raise InferenceError(f"Low-light enhancement failed: {e}")

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        import torch

        try:
            cap = cv2.VideoCapture(str(context.input_path))
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {context.input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # The final stage is encoded with the Export dialog's codec,
            # quality, audio and LUT; intermediate stages stay on the fast
            # built-in writer (see pipeline/video_writer.py).
            sink = open_video_sink(context.output_path, fps=fps, size=(width, height), context=context)

            processed = 0
            device = next(self._model.parameters()).device if not self._is_classical and self._model is not None else None

            context.report_progress(5, f"Processing {total_frames} frames")

            while True:
                if context.abort_flag:
                    raise ProcessingAborted("Video processing aborted")

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    if self._is_classical:
                        output_bgr = self._enhance_classical(frame)
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_tensor = torch.from_numpy(frame_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = self._model(frame_tensor)
                            if isinstance(output, tuple):
                                output = output[0]
                        if self.strength < 1.0:
                            output = frame_tensor * (1 - self.strength) + output * self.strength
                        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        output = (output * 255).clip(0, 255).astype(np.uint8)
                        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                    sink.write(output_bgr)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Frame {processed} low-light enhancement failed: {e}")
                    sink.write(frame)

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
                    'total_frames': total_frames,
                    'strength': self.strength,
                    'fps': fps,
                    'model_type': 'classical' if self._is_classical else 'ai',
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )
        except Exception as e:
            raise InferenceError(f"Video low-light enhancement failed: {e}")


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Low-Light Enhancement",
    pipeline_class=LowLightPipeline,
    available_models=get_registry().get_available_model_names("Low-Light Enhancement"),
    default_model=get_registry().get_default_model("Low-Light Enhancement"),
)
