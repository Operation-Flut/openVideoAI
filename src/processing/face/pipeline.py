"""
Face recovery enhancement pipelines.
Supports CodeFormer, GFPGAN, RestoreFormer, RestoreFormer++, DifFace.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, ModelLoadError, InferenceError, ProcessingAborted
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)


class FaceRecoveryPipeline(EnhancementPipeline):
    """Face recovery enhancement pipeline."""

    config = PipelineConfig(
        name="FaceRecoveryPipeline",
        feature="Face Recovery",
        description="Restore degraded facial details using AI face restoration models",
        supported_formats=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".webm"],
        supports_video=True,
        supports_image=True,
        requires_gpu=True,
        default_parameters={
            "fidelity_weight": 0.5,
            "face_upscale": True,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "fidelity_weight": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "face_upscale": {"type": "boolean", "default": True},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        fidelity_weight: float = 0.5,
        face_upscale: bool = True,
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Face Recovery")
        super().__init__(model_name, device, parameters=parameters, task_id=task_id)
        self.fidelity_weight = max(0.0, min(1.0, fidelity_weight))
        self.face_upscale = face_upscale
        self._model = None
        self._face_helper = None
        # Architecture comes from the models.json catalog entry.
        self._model_config = get_registry().runtime_config(model_name, "Face Recovery")

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Load face recovery model."""
        try:
            import torch

            # Handle both single path and dict of paths
            if isinstance(model_path, dict):
                main_path = model_path.get("main")
            else:
                main_path = model_path

            if not main_path:
                raise ModelLoadError(f"No model file found for {self.model_name}")

            device_obj = torch.device(self.get_device())

            # Load appropriate architecture based on model
            arch = self._model_config['arch']

            if arch == 'CodeFormer':
                from ..archs.codeformer_arch import CodeFormer
                self._model = CodeFormer(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=['32', '64', '128', '256'],
                )
            elif arch == 'GFPGAN':
                # GFPGANv1.4.pth is the "clean" generator, so it strict-loads
                # into the vendored GFPGANv1Clean architecture.
                from ..archs.gfpgan_arch import GFPGANv1Clean
                self._model = GFPGANv1Clean(
                    out_size=512,
                    num_style_feat=512,
                    channel_multiplier=2,
                    decoder_load_path=None,
                    fix_decoder=False,
                    num_mlp=8,
                    input_is_latent=True,
                    different_w=True,
                    narrow=1,
                    sft_half=True,
                )
            elif arch == 'RestoreFormer':
                from ..archs.restoreformer_arch import RestoreFormer
                self._model = RestoreFormer()
            elif arch == 'RestoreFormerPlusPlus':
                # Weights are downloadable but the VQGAN+attention backend is
                # deliberately not bundled; fail fast with a clear message.
                raise ModelLoadError(
                    "RestoreFormer++ backend is not bundled. Implemented face "
                    "models: CodeFormer, GFPGAN, RestoreFormer."
                )
            elif arch == 'DifFace':
                # DifFace uses a different loading mechanism
                raise ModelLoadError("DifFace loading not yet implemented")
            else:
                raise ModelLoadError(f"Unknown architecture: {arch}")

            # Load weights
            state_dict = torch.load(str(main_path), map_location='cpu')
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']

            self._model.load_state_dict(state_dict, strict=True)
            self._model.eval()
            self._model.to(device_obj)

            # Initialize face helper for detection/alignment. FaceRestoreHelper
            # ships with facexlib (gfpgan.utils re-exports it).
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            self._face_helper = FaceRestoreHelper(
                upscale_factor=2 if self.face_upscale else 1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=device_obj,
            )

            self._loaded = True
            logger.info(f"Loaded {self.model_name} ({arch}) on {self.get_device()}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process image or video for face recovery."""
        if not self._model or not self._face_helper:
            raise InferenceError("Model not loaded. Call load_model() first.")

        path = context.input_path
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}

        if is_video:
            return self._process_video(context)
        else:
            return self._process_image(context)

    def _process_image(self, context: ProcessingContext) -> ProcessingResult:
        """Process single image for face recovery."""
        import torch

        try:
            img = cv2.imread(str(context.input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {context.input_path}")

            context.report_progress(10, "Detecting faces")

            # Detect faces
            self._face_helper.clean_all()
            self._face_helper.read_image(img)
            self._face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5
            )
            self._face_helper.align_warp_face()

            if not self._face_helper.cropped_faces:
                logger.warning("No faces detected")
                # Save original if no faces
                cv2.imwrite(str(context.output_path), img)
                context.report_progress(100, "No faces detected, saved original")
                return ProcessingResult(
                    success=True,
                    input_path=context.input_path,
                    output_path=context.output_path,
                    metadata={'faces_detected': 0},
                    model_used=self.model_name,
                    device_used=self.get_device(),
                )

            context.report_progress(30, f"Restoring {len(self._face_helper.cropped_faces)} face(s)")

            # Restore each face
            for idx, cropped_face in enumerate(self._face_helper.cropped_faces):
                # Normalize
                cropped_face_t = torch.from_numpy(cropped_face / 255.0).float().permute(2, 0, 1).unsqueeze(0)
                cropped_face_t = cropped_face_t.to(next(self._model.parameters()).device)

                with torch.no_grad():
                    if self._model_config['arch'] == 'CodeFormer':
                        # CodeFormer returns (output, logits, lq_feat)
                        output, _, _ = self._model(cropped_face_t, w=self.fidelity_weight, adain=True)
                    else:
                        # GFPGAN / RestoreFormer return (image, aux)
                        output, _ = self._model(cropped_face_t)

                    restored_face = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    restored_face = (restored_face * 255).clip(0, 255).astype(np.uint8)

                self._face_helper.add_restored_face(restored_face)

                progress = 30 + int((idx + 1) / len(self._face_helper.cropped_faces) * 50)
                context.report_progress(progress, f"Restored face {idx + 1}/{len(self._face_helper.cropped_faces)}")

            # Paste back
            context.report_progress(85, "Pasting restored faces")
            self._face_helper.get_inverse_affine(None)
            restored_img = self._face_helper.paste_faces_to_input_image()

            # Save
            write_params = []
            if context.output_path.suffix.lower() in {'.jpg', '.jpeg'}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

            if not cv2.imwrite(str(context.output_path), restored_img, write_params):
                raise InferenceError(f"Failed to write output: {context.output_path}")

            context.report_progress(100, "Face recovery complete")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'faces_detected': len(self._face_helper.cropped_faces),
                    'fidelity_weight': self.fidelity_weight,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Face recovery failed: {e}")

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        """Process video for face recovery (frame by frame)."""
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
            total_faces = 0

            context.report_progress(5, f"Processing {total_frames} frames")

            while True:
                if context.abort_flag:
                    raise ProcessingAborted("Video processing aborted")

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    self._face_helper.clean_all()
                    self._face_helper.read_image(frame)
                    self._face_helper.get_face_landmarks_5(
                        only_center_face=False,
                        resize=640,
                        eye_dist_threshold=5
                    )
                    self._face_helper.align_warp_face()

                    if self._face_helper.cropped_faces:
                        total_faces += len(self._face_helper.cropped_faces)
                        for cropped_face in self._face_helper.cropped_faces:
                            cropped_face_t = torch.from_numpy(cropped_face / 255.0).float().permute(2, 0, 1).unsqueeze(0)
                            cropped_face_t = cropped_face_t.to(next(self._model.parameters()).device)

                            with torch.no_grad():
                                if self._model_config['arch'] == 'CodeFormer':
                                    # CodeFormer returns (output, logits, lq_feat)
                                    output, _, _ = self._model(cropped_face_t, w=self.fidelity_weight, adain=True)
                                else:
                                    # GFPGAN / RestoreFormer return (image, aux)
                                    output, _ = self._model(cropped_face_t)

                                restored_face = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                                restored_face = (restored_face * 255).clip(0, 255).astype(np.uint8)

                            self._face_helper.add_restored_face(restored_face)

                        self._face_helper.get_inverse_affine(None)
                        restored_frame = self._face_helper.paste_faces_to_input_image()
                    else:
                        restored_frame = frame

                    out.write(restored_frame)
                    processed += 1

                except Exception as e:
                    logger.warning(f"Frame {processed} face recovery failed: {e}")
                    out.write(frame)

                if processed % 30 == 0 or processed == total_frames:
                    progress = 5 + int((processed / total_frames) * 95)
                    context.report_progress(progress, f"Frame {processed}/{total_frames} ({total_faces} faces restored)")

            cap.release()
            out.release()

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'frames_processed': processed,
                    'total_frames': total_frames,
                    'total_faces_restored': total_faces,
                    'fidelity_weight': self.fidelity_weight,
                    'fps': fps,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )

        except Exception as e:
            raise InferenceError(f"Video face recovery failed: {e}")


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Face Recovery",
    pipeline_class=FaceRecoveryPipeline,
    available_models=get_registry().get_available_model_names("Face Recovery"),
    default_model=get_registry().get_default_model("Face Recovery"),
)