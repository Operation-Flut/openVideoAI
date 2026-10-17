"""
Frame Interpolation / Slow Motion pipelines.

Every model, its runtime architecture and whether it is a classical (non-AI)
optical-flow method come from the ``models.json`` catalog. Classical CPU
fallbacks (Farnebäck, DIS, Lucas-Kanade, and Dual TV-L1 where OpenCV contrib is
installed) need no model download.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, ModelLoadError, InferenceError, ProcessingAborted
from ..pipeline.video_writer import open_video_sink
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)


class FrameInterpolationPipeline(EnhancementPipeline):
    """Frame interpolation and slow motion pipeline."""

    config = PipelineConfig(
        name="FrameInterpolationPipeline",
        feature="Frame Interpolation",
        description="Generate intermediate frames to increase video framerate or create smooth slow motion",
        supported_formats=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"],
        supports_video=True,
        supports_image=False,
        requires_gpu=False,  # Optical flow methods run on CPU
        default_parameters={
            "multiplier": 2,  # 2x, 4x, 8x frame rate
            "slowmo_factor": 1.0,  # 1.0 = normal, 0.5 = 2x slow, 0.25 = 4x slow
            "scene_threshold": 0.3,
            "use_scene_detection": True,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "multiplier": {"type": "integer", "enum": [2, 4, 8], "default": 2},
                "slowmo_factor": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 1.0},
                "scene_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.3},
                "use_scene_detection": {"type": "boolean", "default": True},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        multiplier: int = 2,
        slowmo_factor: float = 1.0,
        scene_threshold: float = 0.3,
        use_scene_detection: bool = True,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Frame Interpolation")
        super().__init__(model_name, device, task_id=task_id)
        self.multiplier = max(1, min(8, multiplier))
        self.slowmo_factor = max(0.1, min(1.0, slowmo_factor))
        self.scene_threshold = scene_threshold
        self.use_scene_detection = use_scene_detection
        self._model = None
        self._flow_model = None
        # Architecture and whether this is a classical (non-AI) method both come
        # from the models.json catalog entry.
        self._model_config = get_registry().runtime_config(model_name, "Frame Interpolation")
        self._is_classical = bool(self._model_config.get('is_classical'))

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load frame interpolation model."""
        if self._is_classical:
            # Classical methods don't need model files
            self._init_classical_flow()
            self._loaded = True
            logger.info(f"Initialized classical flow: {self.model_name}")
            return True

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

            if arch == 'RIFE':
                from ..archs.rife_arch import RIFE
                self._model = RIFE()
                state_dict = torch.load(str(main_path), map_location='cpu')
                self._model.load_state_dict(state_dict, strict=True)
            elif arch == 'FILM':
                from ..archs.film_arch import FILM
                self._model = FILM()
                state_dict = torch.load(str(main_path), map_location='cpu')
                self._model.load_state_dict(state_dict, strict=True)
            elif arch == 'RAFT':
                from ..archs.raft_arch import RAFT
                self._model = RAFT()
                state_dict = torch.load(str(main_path), map_location='cpu')
                self._model.load_state_dict(state_dict, strict=True)
            elif arch == 'FlowFormer':
                from ..archs.flowformer_arch import FlowFormer
                self._model = FlowFormer()
                state_dict = torch.load(str(main_path), map_location='cpu')
                self._model.load_state_dict(state_dict, strict=True)
            elif arch == 'GMA':
                from ..archs.gma_arch import GMA
                self._model = GMA()
                state_dict = torch.load(str(main_path), map_location='cpu')
                self._model.load_state_dict(state_dict, strict=True)
            elif arch == 'SPyNet':
                from ..archs.spynet_arch import SPyNet
                self._model = SPyNet()
                state_dict = torch.load(str(main_path), map_location='cpu')
                self._model.load_state_dict(state_dict, strict=True)
            else:
                raise ModelLoadError(f"Unknown architecture: {arch}")

            self._model.eval()
            self._model.to(device_obj)

            self._loaded = True
            logger.info(f"Loaded {self.model_name} ({arch}) on {self.get_device()}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def _init_classical_flow(self):
        """Initialize the classical optical-flow backend (only where needed)."""
        arch = self._model_config['arch']
        if arch == 'Farneback':
            # ``cv2.calcOpticalFlowFarneback`` is called directly, so there is
            # no persistent handle to create (and no contrib dependency).
            self._flow_model = None
        elif arch == 'DIS':
            self._flow_model = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif arch == 'DualTVL1':
            try:
                self._flow_model = cv2.optflow.DualTVL1OpticalFlow_create()
            except AttributeError as exc:
                raise ModelLoadError(
                    "Dual TV-L1 requires the OpenCV contrib package "
                    "(pip install opencv-contrib-python)"
                ) from exc
        elif arch == 'LucasKanade':
            # Implemented per frame, no persistent model.
            self._flow_model = None
        else:
            raise ModelLoadError(f"Unknown classical flow method: {arch}")

    def _compute_flow(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        """Compute optical flow between two frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        arch = self._model_config['arch']

        if arch == 'Farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        elif arch == 'DIS':
            flow = self._flow_model.calc(prev_gray, next_gray, None)
        elif arch == 'DualTVL1':
            flow = self._flow_model.calc(prev_gray, next_gray, None)
        elif arch == 'LucasKanade':
            # Sparse pyramidal LK flow, densified by inpainting each component
            # separately so the result keeps the expected (h, w, 2) shape.
            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10
            )
            if prev_pts is not None and len(prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None)
                if next_pts is not None:
                    status = status.reshape(-1) if status is not None else np.ones(len(prev_pts), dtype=np.uint8)
                    mask = np.zeros(prev_gray.shape[:2], dtype=np.uint8)
                    for i, (p, n) in enumerate(zip(prev_pts, next_pts)):
                        if not status[i]:
                            continue
                        x, y = int(round(p[0][0])), int(round(p[0][1]))
                        if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                            flow[y, x] = [n[0][0] - p[0][0], n[0][1] - p[0][1]]
                            mask[y, x] = 1
                    if mask.any():
                        holes = (mask == 0).astype(np.uint8)
                        flow[:, :, 0] = cv2.inpaint(flow[:, :, 0], holes, 3, cv2.INPAINT_TELEA)
                        flow[:, :, 1] = cv2.inpaint(flow[:, :, 1], holes, 3, cv2.INPAINT_TELEA)
        else:
            raise InferenceError(f"Unknown classical flow method: {arch}")

        return flow

    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
        """Warp frame using optical flow at time t."""
        h, w = frame.shape[:2]
        # Create grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        # Apply flow scaled by t
        map_x = (grid_x + flow[:, :, 0] * t).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1] * t).astype(np.float32)
        # Remap
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return warped

    def _detect_scene_change(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> bool:
        """Detect scene change using histogram difference."""
        prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        next_hist = cv2.calcHist([next_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(prev_hist, prev_hist)
        cv2.normalize(next_hist, next_hist)
        correlation = cv2.compareHist(prev_hist, next_hist, cv2.HISTCMP_CORREL)
        return correlation < (1 - self.scene_threshold)

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process video for frame interpolation / slow motion."""
        if not self._loaded:
            raise InferenceError("Model not loaded. Call load_model() first.")

        return self._process_video(context)

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        """Process video for frame interpolation."""
        import torch

        try:
            cap = cv2.VideoCapture(str(context.input_path))
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {context.input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate output FPS
            if self.slowmo_factor < 1.0:
                # Slow motion: multiply frame count, keep same duration
                out_fps = fps
                frame_multiplier = int(1 / self.slowmo_factor)
            else:
                # Frame rate increase
                out_fps = fps * self.multiplier
                frame_multiplier = self.multiplier

            # The final stage is encoded with the Export dialog's codec,
            # quality, audio and LUT; intermediate stages stay on the fast
            # built-in writer (see pipeline/video_writer.py).
            sink = open_video_sink(context.output_path, fps=out_fps, size=(width, height), context=context)

            context.report_progress(5, f"Reading first frame")

            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                raise InferenceError("Empty video")

            sink.write(prev_frame)
            written_frames = 1
            processed_pairs = 0

            device = None
            if self._model is not None:
                device = next(self._model.parameters()).device

            total_pairs = total_frames - 1
            context.report_progress(10, f"Interpolating {total_pairs} frame pairs")

            while True:
                if context.abort_flag:
                    raise ProcessingAborted("Video processing aborted")

                ret, next_frame = cap.read()
                if not ret:
                    break

                # Check for scene change
                if self.use_scene_detection and self._detect_scene_change(prev_frame, next_frame):
                    # Scene change - just write next frame without interpolation
                    sink.write(next_frame)
                    written_frames += 1
                else:
                    # Interpolate frames between prev and next
                    if self._is_classical:
                        flow = self._compute_flow(prev_frame, next_frame)
                        for i in range(1, frame_multiplier):
                            t = i / frame_multiplier
                            interp_frame = self._warp_frame(prev_frame, flow, t)
                            sink.write(interp_frame)
                            written_frames += 1
                    else:
                        # AI-based interpolation
                        with torch.no_grad():
                            prev_tensor = torch.from_numpy(prev_frame / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
                            next_tensor = torch.from_numpy(next_frame / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

                            for i in range(1, frame_multiplier):
                                t = i / frame_multiplier
                                # Different models have different interfaces
                                if hasattr(self._model, 'interpolate'):
                                    interp_tensor = self._model.interpolate(prev_tensor, next_tensor, t)
                                else:
                                    # RIFE-style: infer flow then warp
                                    flow = self._model(torch.cat([prev_tensor, next_tensor], dim=1))
                                    interp_tensor = self._warp_tensor(prev_tensor, flow, t)

                                interp_frame = interp_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                                interp_frame = (interp_frame * 255).clip(0, 255).astype(np.uint8)
                                sink.write(interp_frame)
                                written_frames += 1

                    # Write the next frame
                    sink.write(next_frame)
                    written_frames += 1

                processed_pairs += 1

                if processed_pairs % 10 == 0 or processed_pairs == total_pairs:
                    progress = 10 + int((processed_pairs / total_pairs) * 85)
                    context.report_progress(progress, f"Processed {processed_pairs}/{total_pairs} pairs ({written_frames} frames written)")

                prev_frame = next_frame

            cap.release()
            sink.close()

            context.report_progress(100, f"Complete: {written_frames} frames at {out_fps:.1f}fps")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'input_fps': fps,
                    'output_fps': out_fps,
                    'input_frames': total_frames,
                    'output_frames': written_frames,
                    'multiplier': frame_multiplier,
                    'slowmo_factor': self.slowmo_factor,
                    'model_type': 'classical' if self._is_classical else 'ai',
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Frame interpolation failed: {e}")

    def _warp_tensor(self, frame_tensor, flow, t):
        """Warp tensor using flow (for AI models)."""
        # Simplified - actual implementation depends on model
        import torch
        import torch.nn.functional as F

        _, _, h, w = frame_tensor.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=frame_tensor.device),
            torch.arange(w, device=frame_tensor.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float().unsqueeze(0)

        flow_scaled = flow * t
        grid_warped = grid + flow_scaled

        # Normalize to [-1, 1]
        grid_warped[:, 0] = 2 * grid_warped[:, 0] / (w - 1) - 1
        grid_warped[:, 1] = 2 * grid_warped[:, 1] / (h - 1) - 1
        grid_warped = grid_warped.permute(0, 2, 3, 1)

        warped = F.grid_sample(frame_tensor, grid_warped, mode='bilinear', padding_mode='reflection', align_corners=True)
        return warped


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Frame Interpolation",
    pipeline_class=FrameInterpolationPipeline,
    available_models=get_registry().get_available_model_names("Frame Interpolation"),
    default_model=get_registry().get_default_model("Frame Interpolation"),
)