"""
Color Transfer / Match Color pipelines.
Models, architectures and defaults come from the models.json catalog.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, ModelLoadError, InferenceError, UnsupportedFormatError, ProcessingAborted
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)


class MatchColorPipeline(EnhancementPipeline):
    """Color transfer / match color pipeline."""

    config = PipelineConfig(
        name="MatchColorPipeline",
        feature="Match Color",
        description="Transfer color grading from a reference image to target media",
        supported_formats=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".webm"],
        supports_video=True,
        supports_image=True,
        requires_gpu=False,
        default_parameters={
            "strength": 1.0,
            "preserve_luminance": True,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
                "preserve_luminance": {"type": "boolean", "default": True},
                "reference_path": {"type": "string", "description": "Path to reference image"},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        strength: float = 1.0,
        preserve_luminance: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Match Color")
        super().__init__(model_name, device, parameters=parameters, task_id=task_id)
        self.strength = max(0.0, min(1.0, strength))
        self.preserve_luminance = preserve_luminance
        self._model = None
        self._reference_path: Optional[str] = None
        # Architecture comes from the models.json catalog entry.
        self._model_config = get_registry().runtime_config(model_name, "Match Color")

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load color transfer model."""
        arch = self._model_config['arch']

        if arch == 'Reinhard':
            # No model file needed for Reinhard
            self._loaded = True
            logger.info("Initialized Reinhard Transfer (no model file needed)")
            return True

        try:
            import torch

            if isinstance(model_path, dict):
                main_path = model_path.get("main")
                additional = model_path.get("additional", {})
            else:
                main_path = model_path
                additional = {}

            if not main_path:
                raise ModelLoadError(f"No model file found for {self.model_name}")

            device_obj = torch.device(self.get_device())

            if arch == 'WCT2':
                # archs/wct2_arch ships the complete WCT2 network (wavelet
                # encoder/decoder plus the whitening/colouring transform), so use
                # it directly instead of re-deriving the transform here.
                from ..archs.wct2_arch import WCT2

                decoder_path = additional.get('wave_decoder_cat5_l4.pth')
                if not decoder_path:
                    # Try to find decoder in same directory
                    decoder_path = Path(main_path).parent / 'wave_decoder_cat5_l4.pth'

                if not Path(decoder_path).exists():
                    raise ModelLoadError(f"WCT2 decoder checkpoint not found: {decoder_path}")

                # The wavelet unpooling in the decoder is a depthwise
                # ConvTranspose2d, which torch_directml cannot execute
                # ("Falscher Parameter" / ERROR_INVALID_PARAMETER at runtime), so
                # this architecture runs on the CPU even when a GPU is selected.
                device_obj = torch.device('cpu')
                if self.get_device() != 'cpu':
                    logger.info(
                        "WCT2 runs on the CPU: DirectML does not support the "
                        "depthwise transposed convolutions its decoder needs."
                    )

                self._model = WCT2(device='cpu')
                self._model.load_weights(str(main_path), str(decoder_path))

            elif arch == 'NLUT':
                # NLUT checkpoints are training checkpoints
                # ({iter, state_dict, optimizer}) that also contain auxiliary
                # modules outside the inference graph; from_checkpoint unwraps
                # them and builds the matching 2048+32+32 / dim 33 network.
                from ..archs.nlut_arch import NLUTNet

                self._model = NLUTNet.from_checkpoint(
                    str(main_path), device=str(device_obj), strict=False,
                )

            elif arch == 'DeepColorTransfer':
                from ..archs.deepcolor_arch import DeepColorTransferGenerator
                self._model = DeepColorTransferGenerator()
                state_dict = torch.load(str(main_path), map_location='cpu')
                if 'params' in state_dict:
                    state_dict = state_dict['params']
                self._model.load_state_dict(state_dict, strict=True)
                self._model.eval()
                self._model.to(device_obj)

            else:
                raise ModelLoadError(f"Unknown architecture: {arch}")

            self._loaded = True
            logger.info(f"Loaded {self.model_name} ({arch}) on {self.get_device()}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def _reinhard_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Classic Reinhard color transfer (mean/std in LAB space)."""
        # Convert to LAB
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute statistics
        src_mean, src_std = cv2.meanStdDev(source_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

        src_mean = src_mean.flatten()
        src_std = src_std.flatten()
        tgt_mean = tgt_mean.flatten()
        tgt_std = tgt_std.flatten()

        # Avoid division by zero
        src_std[src_std == 0] = 1

        # Transfer
        result_lab = (target_lab - tgt_mean) * (src_std / tgt_std) + src_mean

        # Blend with original based on strength
        if self.strength < 1.0:
            result_lab = target_lab * (1 - self.strength) + result_lab * self.strength

        # Preserve luminance if requested
        if self.preserve_luminance:
            result_lab[:, :, 0] = target_lab[:, :, 0]

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        return result

    @staticmethod
    def _pad_to_multiple(tensor, multiple: int = 8):
        """Reflect-pad H/W up to a multiple of ``multiple``.

        WCT2 halves the resolution three times (wavelet pooling), so its inputs
        must be a multiple of 8 for the skip connections to line up.
        """
        import torch.nn.functional as F

        h, w = tensor.shape[-2:]
        pad_h, pad_w = (-h) % multiple, (-w) % multiple
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        return tensor

    def _wct2_transfer(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
        """WCT2 wavelet colour transfer.

        Delegates to the whitening/colouring transform implemented in
        ``archs/wct2_arch`` (``WCT2.transfer`` -> ``feature_wct``/``wct_core``).
        """
        import torch

        device = next(self._model.parameters()).device

        # Preprocess
        content_tensor = torch.from_numpy(content / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
        style_tensor = torch.from_numpy(style / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        content_tensor = self._pad_to_multiple(content_tensor)
        style_tensor = self._pad_to_multiple(style_tensor)

        with torch.no_grad():
            result = self._model.transfer(content_tensor, style_tensor, alpha=self.strength)

        # Crop the padding back off (H/W as in the original content image)
        result = result.squeeze(0)[:, :content.shape[0], :content.shape[1]]
        result = result.permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return result

    def _nlut_transfer(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
        """NLUT color transfer."""
        import torch

        device = next(self._model.parameters()).device

        content_tensor = torch.from_numpy(content / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
        style_tensor = torch.from_numpy(style / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            # NLUTNet.forward(img, img_org, style): features come from ``img``,
            # the predicted 3D LUT is applied to ``img_org`` and the result is
            # returned as (image, image, {"LUT": ..., "tvmn": ...}).
            result, _, _ = self._model(content_tensor, content_tensor, style_tensor)

        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return result

    def _deep_transfer(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
        """Deep Color Transfer."""
        import torch

        device = next(self._model.parameters()).device

        content_tensor = torch.from_numpy(content / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
        style_tensor = torch.from_numpy(style / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            result = self._model(content_tensor, style_tensor)

        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return result

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process image or video for color transfer."""
        if not self._loaded and self._model_config['arch'] != 'Reinhard':
            raise InferenceError("Model not loaded. Call load_model() first.")

        # Get reference image path. The chain merges each pipeline's parameters
        # into the context, but fall back to the instance's own parameters so a
        # directly constructed pipeline works too.
        reference_path = context.parameters.get('reference_path') or self.parameters.get('reference_path')
        if not reference_path:
            raise UnsupportedFormatError("Reference image required for color transfer")

        ref_path = Path(reference_path)
        if not ref_path.exists():
            raise UnsupportedFormatError(f"Reference image not found: {reference_path}")
        self._reference_path = str(ref_path)

        # Load reference
        reference = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise UnsupportedFormatError(f"Failed to read reference image: {reference_path}")

        path = context.input_path
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

        if is_video:
            return self._process_video(context, reference)
        else:
            return self._process_image(context, reference)

    def _process_image(self, context: ProcessingContext, reference: np.ndarray) -> ProcessingResult:
        """Process single image for color transfer."""
        try:
            img = cv2.imread(str(context.input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {context.input_path}")

            context.report_progress(10, "Running color transfer")

            arch = self._model_config['arch']

            if arch == 'Reinhard':
                result = self._reinhard_transfer(reference, img)
            elif arch == 'WCT2':
                result = self._wct2_transfer(img, reference)
            elif arch == 'NLUT':
                result = self._nlut_transfer(img, reference)
            elif arch == 'DeepColorTransfer':
                result = self._deep_transfer(img, reference)
            else:
                raise InferenceError(f"Unknown color transfer method: {arch}")

            context.report_progress(80, "Saving output")

            write_params = []
            if context.output_path.suffix.lower() in {'.jpg', '.jpeg'}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

            if not cv2.imwrite(str(context.output_path), result, write_params):
                raise InferenceError(f"Failed to write output: {context.output_path}")

            context.report_progress(100, "Color transfer complete")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'strength': self.strength,
                    'preserve_luminance': self.preserve_luminance,
                    'reference_path': self._reference_path,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Color transfer failed: {e}")

    def _process_video(self, context: ProcessingContext, reference: np.ndarray) -> ProcessingResult:
        """Process video for color transfer."""
        try:
            cap = cv2.VideoCapture(str(context.input_path))
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {context.input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Resize reference to match video resolution
            reference = cv2.resize(reference, (width, height))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(context.output_path), fourcc, fps, (width, height))

            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {context.output_path}")

            processed = 0
            arch = self._model_config['arch']

            context.report_progress(5, f"Processing {total_frames} frames")

            while True:
                if context.abort_flag:
                    raise ProcessingAborted("Video processing aborted")

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    if arch == 'Reinhard':
                        result = self._reinhard_transfer(reference, frame)
                    elif arch == 'WCT2':
                        result = self._wct2_transfer(frame, reference)
                    elif arch == 'NLUT':
                        result = self._nlut_transfer(frame, reference)
                    elif arch == 'DeepColorTransfer':
                        result = self._deep_transfer(frame, reference)
                    else:
                        result = frame

                    out.write(result)
                    processed += 1

                except Exception as e:
                    logger.warning(f"Frame {processed} color transfer failed: {e}")
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
                    'preserve_luminance': self.preserve_luminance,
                    'fps': fps,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Video color transfer failed: {e}")


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Match Color",
    pipeline_class=MatchColorPipeline,
    available_models=get_registry().get_available_model_names("Match Color"),
    default_model=get_registry().get_default_model("Match Color"),
)