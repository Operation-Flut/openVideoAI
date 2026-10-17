"""
Video enhancement pipelines.
RealBasicVSR (multi-frame video super-resolution) and TSRN-RTVD (real-time video
deblurring). Models, architectures and defaults come from the models.json catalog.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from collections import deque

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, ModelLoadError, InferenceError, ProcessingAborted
from ..pipeline.video_writer import open_video_sink
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)


class RealBasicVSRPipeline(EnhancementPipeline):
    """RealBasicVSR video super-resolution pipeline."""

    config = PipelineConfig(
        name="RealBasicVSRPipeline",
        feature="Upscale",
        description="Multi-frame video super-resolution with temporal consistency",
        supported_formats=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"],
        supports_video=True,
        supports_image=False,
        requires_gpu=True,
        default_parameters={
            "scale": 4,
            "tile_size": 0,  # Process full frames for video models
            "fp16": True,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "scale": {"type": "integer", "enum": [2, 4], "default": 4},
                "fp16": {"type": "boolean", "default": True},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        scale: int = 4,
        fp16: bool = True,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Upscale")
        super().__init__(model_name, device, task_id=task_id)
        self.scale = scale
        self.fp16 = fp16
        self._model = None
        self._spynet = None  # For optical flow
        # Architecture and tunables come from the models.json catalog entry.
        self._model_config = get_registry().runtime_config(model_name, "Upscale")

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load RealBasicVSR model."""
        try:
            import torch

            if isinstance(model_path, dict):
                main_path = model_path.get("main")
            else:
                main_path = model_path

            if not main_path:
                raise ModelLoadError(f"No model file found for {self.model_name}")

            device_obj = torch.device(self.get_device())

            # Load SPyNet for optical flow (required by RealBasicVSR)
            from ..archs.spynet_arch import SPyNet
            self._spynet = SPyNet()
            # SPyNet weights would need to be downloaded separately
            # For now, we'll use a pre-trained one or skip if not available

            # Load RealBasicVSR
            from ..archs.realbasicvsr_arch import RealBasicVSR
            self._model = RealBasicVSR(
                num_feat=64,
                num_block=10,
                spynet_path=None,  # Will use internal flow
                cpu_cache_length=100,
            )

            state_dict = torch.load(str(main_path), map_location='cpu')
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']

            self._model.load_state_dict(state_dict, strict=True)
            self._model.eval()
            self._model.to(device_obj)

            if self.fp16 and device_obj.type in ('cuda', 'hip', 'dml', 'privateuseone'):
                self._model.half()

            self._loaded = True
            logger.info(f"Loaded {self.model_name} on {self.get_device()}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process video with RealBasicVSR."""
        if not self._model:
            raise InferenceError("Model not loaded. Call load_model() first.")

        return self._process_video(context)

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        """Process video with multi-frame super-resolution."""
        import torch

        try:
            cap = cv2.VideoCapture(str(context.input_path))
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {context.input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_width = width * self.scale
            out_height = height * self.scale

            # The final stage is encoded with the Export dialog's codec,
            # quality, audio and LUT; intermediate stages stay on the fast
            # built-in writer (see pipeline/video_writer.py).
            sink = open_video_sink(context.output_path, fps=fps, size=(out_width, out_height), context=context)

            device = next(self._model.parameters()).device
            processed = 0

            # RealBasicVSR processes frames sequentially with temporal context
            # It maintains an internal state of previous frames
            context.report_progress(5, f"Processing {total_frames} frames with temporal context")

            # Read all frames first (for sliding window approach)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            if not frames:
                raise InferenceError("Empty video")

            # Process with sliding window
            # RealBasicVSR typically uses 3-5 frames as context
            window_size = 5
            half_window = window_size // 2

            with torch.no_grad():
                for i in range(len(frames)):
                    if context.abort_flag:
                        raise ProcessingAborted("Video processing aborted")

                    # Build window around frame i
                    window_frames = []
                    for j in range(i - half_window, i + half_window + 1):
                        idx = max(0, min(len(frames) - 1, j))
                        frame = frames[idx]
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_tensor = torch.from_numpy(frame_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
                        window_frames.append(frame_tensor)

                    # Stack: (1, T, C, H, W)
                    input_tensor = torch.stack(window_frames, dim=1).to(device)
                    if self.fp16 and device.type in ('cuda', 'hip', 'dml', 'privateuseone'):
                        input_tensor = input_tensor.half()

                    # Forward pass
                    output_tensor = self._model(input_tensor)
                    # Output shape: (1, C, H*scale, W*scale)
                    output_tensor = output_tensor.squeeze(0).squeeze(0)  # (C, H, W)

                    # Convert back
                    output = output_tensor.permute(1, 2, 0).cpu().numpy()
                    output = (output * 255).clip(0, 255).astype(np.uint8)
                    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                    sink.write(output_bgr)
                    processed += 1

                    if processed % 10 == 0 or processed == total_frames:
                        progress = 5 + int((processed / total_frames) * 95)
                        context.report_progress(progress, f"Frame {processed}/{total_frames}")

            sink.close()

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'frames_processed': processed,
                    'total_frames': total_frames,
                    'input_resolution': f"{width}x{height}",
                    'output_resolution': f"{out_width}x{out_height}",
                    'scale_factor': self.scale,
                    'fps': fps,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"RealBasicVSR processing failed: {e}")


class TSRNRTVDPipeline(EnhancementPipeline):
    """TSRN-RTVD real-time video deblurring pipeline."""

    config = PipelineConfig(
        name="TSRNRTVDPipeline",
        feature="Sharpen / Deblur",
        description="Real-time video deblurring with temporal spatial residual network",
        supported_formats=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"],
        supports_video=True,
        supports_image=True,
        requires_gpu=False,
        default_parameters={
            "reset_interval": 10,  # Reset hidden state every N frames
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "reset_interval": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        reset_interval: int = 10,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Sharpen / Deblur")
        super().__init__(model_name, device, task_id=task_id)
        self.reset_interval = reset_interval
        self._model = None
        self._hidden_state = None
        self._frame_count = 0
        # Architecture comes from the models.json catalog entry.
        self._model_config = get_registry().runtime_config(model_name, "Sharpen / Deblur")

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load TSRN-RTVD model."""
        try:
            import torch

            if isinstance(model_path, dict):
                main_path = model_path.get("main")
            else:
                main_path = model_path

            if not main_path:
                raise ModelLoadError(f"No model file found for {self.model_name}")

            device_obj = torch.device(self.get_device())

            from ..archs.tsrn_arch import TSRN
            self._model = TSRN()
            state_dict = torch.load(str(main_path), map_location='cpu')
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']

            self._model.load_state_dict(state_dict, strict=True)
            self._model.eval()
            self._model.to(device_obj)

            self._loaded = True
            logger.info(f"Loaded {self.model_name} on {self.get_device()}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def reset_state(self):
        """Reset hidden state for streaming processing."""
        self._hidden_state = None
        self._frame_count = 0

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process image or video with TSRN-RTVD."""
        if not self._model:
            raise InferenceError("Model not loaded. Call load_model() first.")

        path = context.input_path
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

        if is_video:
            return self._process_video(context)
        else:
            return self._process_image(context)

    def _process_image(self, context: ProcessingContext) -> ProcessingResult:
        """Process single image (treated as single-frame video)."""
        import torch

        try:
            img = cv2.imread(str(context.input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {context.input_path}")

            context.report_progress(10, "Processing image")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(1)  # (1, 1, C, H, W)
            device = next(self._model.parameters()).device
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                output, self._hidden_state = self._model(img_tensor, self._hidden_state)

            output = output.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = (output * 255).clip(0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            write_params = []
            if context.output_path.suffix.lower() in {'.jpg', '.jpeg'}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

            if not cv2.imwrite(str(context.output_path), output_bgr, write_params):
                raise InferenceError(f"Failed to write output: {context.output_path}")

            context.report_progress(100, "Deblurring complete")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={'model_arch': 'TSRN'},
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Image deblurring failed: {e}")

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        """Process video with streaming TSRN-RTVD."""
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

            device = next(self._model.parameters()).device
            processed = 0

            # Reset state at start
            self.reset_state()

            context.report_progress(5, f"Processing {total_frames} frames (streaming)")

            with torch.no_grad():
                while True:
                    if context.abort_flag:
                        raise ProcessingAborted("Video processing aborted")

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(device)

                    output, self._hidden_state = self._model(frame_tensor, self._hidden_state)

                    output = output.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
                    output = (output * 255).clip(0, 255).astype(np.uint8)
                    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                    sink.write(output_bgr)
                    processed += 1
                    self._frame_count += 1

                    # Reset hidden state periodically to avoid drift
                    if self._frame_count % self.reset_interval == 0:
                        self._hidden_state = None

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
                    'reset_interval': self.reset_interval,
                    'fps': fps,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Video deblurring failed: {e}")


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Upscale",
    pipeline_class=RealBasicVSRPipeline,
    available_models=get_registry().get_model_names_by_arch("RealBasicVSR"),
    default_model=get_registry().get_default_model_for_arch("Upscale", "RealBasicVSR"),
)

register_pipeline(
    feature="Sharpen / Deblur",
    pipeline_class=TSRNRTVDPipeline,
    available_models=get_registry().get_model_names_by_arch("TSRN"),
    default_model=get_registry().get_default_model_for_arch("Sharpen / Deblur", "TSRN"),
)