"""
Face Recovery processing using CodeFormer
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..distutils_compat import install_distutils_compat

install_distutils_compat()

from .base import (
    EnhancementPipeline,
    ModelLoadError,
    InferenceError,
    ProcessingAborted,
)

logger = logging.getLogger(__name__)

# Face Recovery Model Configurations
FACE_MODELS = {
    'CodeFormer': {
        'model_file': 'codeformer.pth',
        'description': 'Advanced face restoration for severely degraded or blurred faces',
        'fidelity_weight': 0.5,
        'tile_size': 512,
    },
    'GFPGAN': {
        'model_file': 'GFPGANv1.4.pth',
        'description': 'Face restoration with GFP-GAN architecture (v1.4, clean)',
        'fidelity_weight': 0.5,
        'tile_size': 512,
    },
    'RestoreFormer': {
        'model_file': 'RestoreFormer.pth',
        'description': 'High-quality face restoration using a VQGAN + multi-head attention transformer',
        'fidelity_weight': 0.5,
        'tile_size': 512,
    },
    'RestoreFormer++': {
        'model_file': 'RestoreFormer++.ckpt',
        'description': 'Upgraded RestoreFormer with a large VQGAN backbone for finer skin texture',
        'fidelity_weight': 0.5,
        'tile_size': 512,
    }
}

# Face models whose inference pipeline is wired up in this build. Everything
# else in FACE_MODELS is registered (weights downloadable) but fails fast with
# a clear message until its backend is implemented.
_IMPLEMENTED_FACE_MODELS = 'CodeFormer, GFPGAN, RestoreFormer'

# Architecture configuration of the official CodeFormer checkpoint
# (codeformer.pth v0.1.0). These hyperparameters match the released weights
# so loading is strict.
_CODEFORMER_KWARGS = dict(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=['32', '64', '128', '256'],
)

# Architecture configuration of the official GFPGAN v1.4 checkpoint
# (GFPGANv1.4.pth). Matches the defaults used by GFPGANer for arch='clean'.
_GFPGAN_KWARGS = dict(
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

# Architecture configuration of the official RestoreFormer checkpoint
# (RestoreFormer.pth, GFPGAN release v1.3.0). These are the exact defaults
# used by GFPGANer for arch='restoreformer'.
_RESTOREFORMER_KWARGS = dict(
    n_embed=1024,
    embed_dim=256,
    ch=64,
    out_ch=3,
    ch_mult=(1, 2, 2, 4, 4, 8),
    num_res_blocks=2,
    attn_resolutions=(16,),
    dropout=0.0,
    in_channels=3,
    resolution=512,
    z_channels=256,
    double_z=False,
    enable_mid=True,
    head_size=8,
)

# Face detector used by facexlib inside the CodeFormer pipeline
_FACE_DET_MODEL = 'retinaface_resnet50'


class FaceRecoveryPipeline(EnhancementPipeline):
    """Face recovery enhancement pipeline"""

    def __init__(self, model_name: str = 'CodeFormer', device: str = 'auto',
                 fidelity_weight: float = 0.5, tile_size: int = 512):
        super().__init__(device)
        self.model_name = model_name
        self.fidelity_weight = fidelity_weight
        self.tile_size = tile_size
        self._model = None
        self._face_helper = None
        self._model_config = FACE_MODELS.get(model_name, FACE_MODELS['CodeFormer'])

    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load the face restoration model (CodeFormer, GFPGAN or RestoreFormer)."""
        try:
            import torch
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            if self.model_name == 'CodeFormer':
                from .archs.codeformer_arch import CodeFormer
                net = CodeFormer(**_CODEFORMER_KWARGS)
            elif self.model_name == 'GFPGAN':
                from .archs.gfpgan_arch import GFPGANv1Clean
                net = GFPGANv1Clean(**_GFPGAN_KWARGS)
            elif self.model_name == 'RestoreFormer':
                from .archs.restoreformer_arch import RestoreFormer
                net = RestoreFormer(**_RESTOREFORMER_KWARGS)
            else:
                raise ModelLoadError(
                    f"Face model '{self.model_name}' is not implemented in this build. "
                    f"Available: {_IMPLEMENTED_FACE_MODELS}"
                )

            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                state = checkpoint.get('params_ema') or checkpoint.get('params')
                if state is None and self.model_name == 'RestoreFormer':
                    # The official RestoreFormer release ships a bare state dict
                    # (GFPGANer falls back to the whole checkpoint as well).
                    state = checkpoint
                if state is None:
                    raise ModelLoadError(
                        f"Checkpoint {model_path} contains neither 'params_ema' nor 'params'."
                    )
                # Legacy CodeFormer checkpoints store parameter names with a
                # double underscore separator instead of dots.
                if any('__' in key for key in state):
                    state = {key.replace('__', '.'): value for key, value in state.items()}
                net.load_state_dict(state)
            else:
                logger.warning('%s checkpoint not found at %s - using random weights',
                               self.model_name, model_path)

            self._model = net.to(self.device)
            self._model.eval()

            self._face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model=_FACE_DET_MODEL,
                save_ext='png',
                use_parse=False,
                device=self.device,
            )

            logger.info("Loaded %s face restoration model on %s",
                        self.model_name, self.device)
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process image or video for face recovery"""
        path = Path(input_path)
        if not path.exists():
            raise InferenceError(f"Input file not found: {input_path}")

        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        is_video = path.suffix.lower() in video_extensions

        if is_video:
            return self._process_video(input_path, output_path, **kwargs)
        else:
            return self._process_image(input_path, output_path, **kwargs)

    def _process_image(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process single image for face recovery"""
        import torch

        progress_callback = kwargs.get('progress_callback')

        try:
            if progress_callback:
                progress_callback(10)

            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")

            if progress_callback:
                progress_callback(20)

            if self._face_helper is None or self._model is None:
                raise InferenceError("Face model is not loaded - call load_model() first.")

            # Detect faces and warp them to the aligned face template
            self._face_helper.clean_all()
            self._face_helper.read_image(img)
            detected = self._face_helper.get_face_landmarks_5(
                resize=640, eye_dist_threshold=5)
            if detected > 0:
                self._face_helper.align_warp_face()
            num_faces = len(self._face_helper.cropped_faces)

            if progress_callback:
                progress_callback(45)

            # Restore each aligned face
            restored_faces = []
            for cropped_face in self._face_helper.cropped_faces:
                # BGR -> RGB and normalize to [-1, 1] like the official
                # CodeFormer/GFPGAN inference code.
                face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                face = (face - 0.5) / 0.5
                face_t = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if self.model_name == 'GFPGAN':
                        # randomize_noise=False -> deterministic, uses stored noise
                        output = self._model(face_t, return_rgb=True, randomize_noise=False)[0]
                    elif self.model_name == 'RestoreFormer':
                        output = self._model(face_t)[0]
                    else:
                        output = self._model(face_t, w=self.fidelity_weight, adain=True)[0]
                    output = output.clamp(-1, 1)

                restored = (output + 1.0) / 2.0
                restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
                restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                restored_faces.append(restored)

            if progress_callback:
                progress_callback(75)

            # Paste the restored faces back onto the input image
            for restored in restored_faces:
                self._face_helper.add_restored_face(restored)
            self._face_helper.get_inverse_affine()
            output_img = self._face_helper.paste_faces_to_input_image()

            if progress_callback:
                progress_callback(90)

            if not cv2.imwrite(output_path, output_img):
                raise InferenceError(f"Failed to write output image: {output_path}")

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'input_shape': img.shape[:2],
                'output_shape': output_img.shape[:2],
                'model': self.model_name,
                'faces_detected': num_faces,
                'faces_restored': len(restored_faces),
            }

        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Face recovery failed: {e}")

    def _process_video(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process video frame by frame"""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")

            processed = 0
            failed = 0
            faces_detected = 0

            progress_callback = kwargs.get('progress_callback')

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Process frame for face recovery
                    temp_input = f"_temp_face_frame_{processed}.jpg"
                    temp_output = f"_temp_face_restored_{processed}.jpg"

                    cv2.imwrite(temp_input, frame)
                    result = self._process_image(temp_input, temp_output, **{})

                    if result.get('success'):
                        restored = cv2.imread(temp_output)
                        out.write(restored)
                        faces_detected += result.get('faces_detected', 0)
                        processed += 1

                        # Cleanup temp files
                        try:
                            Path(temp_input).unlink()
                            Path(temp_output).unlink()
                        except:
                            pass
                    else:
                        failed += 1
                        out.write(frame)

                except Exception as e:
                    logger.warning(f"Frame {processed} failed: {e}")
                    failed += 1
                    out.write(frame)

                if progress_callback:
                    progress_callback(processed / total_frames * 100)

            cap.release()
            out.release()

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': processed,
                'frames_failed': failed,
                'total_frames': total_frames,
                'faces_detected': faces_detected,
                'model': self.model_name
            }

        except ProcessingAborted:
            try:
                cap.release()
            except:
                pass
            try:
                out.release()
            except:
                pass
            raise
        except Exception as e:
            raise InferenceError(f"Face recovery video failed: {e}")


def get_face_model(model_name: str, models_dir: Optional[str] = None) -> Optional[str]:
    """Get path to face recovery model file.

    ``models_dir`` defaults to the application's resolved model directory
    (``src.core.paths.models_dir()``), which is CWD-independent; a plain
    relative ``"models"`` value is still honoured for backwards compatibility.
    """
    model_config = FACE_MODELS.get(model_name)
    if not model_config:
        return None

    filename = model_config.get('model_file')
    if not filename:
        return None

    if models_dir is None:
        try:
            from ..core import paths as _paths
            models_dir = str(_paths.models_dir())
        except Exception:
            models_dir = "models"

    path = Path(models_dir) / filename
    if path.exists():
        return str(path)
    if str(models_dir) != "models":
        legacy = Path("models") / filename
        if legacy.exists():
            return str(legacy)
    return None


def recover_face(input_path: str, output_path: str, model_name: str = 'CodeFormer',
                 device: str = 'auto', models_dir: Optional[str] = None,
                 progress_callback: Optional[callable] = None,
                 fidelity_weight: float = 0.5) -> Dict[str, Any]:
    """Convenience function for face recovery"""
    model_path = get_face_model(model_name, models_dir)

    pipeline = FaceRecoveryPipeline(
        model_name=model_name, device=device, fidelity_weight=fidelity_weight
    )
    pipeline.load_model(model_path)
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def list_face_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """List available face recovery models"""
    results = []
    for name, config in FACE_MODELS.items():
        path = get_face_model(name, models_dir)
        results.append({
            'name': name,
            'description': config['description'],
            'downloaded': path is not None,
            'path': path
        })
    return results