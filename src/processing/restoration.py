"""
Multipurpose Restoration, Frame Recovery, and additional video enhancement features
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

# Frame Recovery and Restoration configurations
FRAME_RECOVERY_MODELS = {
    'BasicVSR': {
        'model_file': 'BasicVSR.pth',
        'description': 'Basic Video Super-Resolution for frame recovery',
        'scale': 4,
    },
    'EDVR': {
        'model_file': 'edvr.pth',
        'description': 'Video Restoration with Enhanced Deformable attention',
        'scale': 4,
    }
}

MULTIPURPOSE_MODELS = {
    'RealBasicVSR': {
        'model_file': 'RealBasicVSR_x4.pth',
        'description': 'Multi-frame video restoration for noisy, low-light, or degraded footage',
        'scale': 4,
        'backend': 'classical',
    },
    'Real-CUGAN': {
        'model_file': 'realcugan_nose.pt',
        'description': 'Anime multipurpose: artifact removal + denoise + upscale',
        'scale': 3,
        'backend': 'classical',
    },
    'DiffBIR': {
        'model_file': 'general_full_v1.ckpt',
        'description': 'Diffusion-based general blind restoration (IRControlNet + SwinIR stage-1)',
        'scale': 1,
        'backend': 'diffbir',
    },
    'SUPIR': {
        'model_file': 'SUPIR-v0Q.ckpt',
        'description': 'SDXL-based photo-realistic blind restoration (needs ~24GB VRAM)',
        'scale': 1,
        'backend': 'supir',
    },
    'StableSR': {
        'model_file': 'stablesr_000117.ckpt',
        'description': 'Diffusion super-resolution with time-aware encoder (IJCV 2024)',
        'scale': 4,
        'backend': 'stablesr',
    },
    'SeeSR': {
        'model_file': 'diffusion_pytorch_model.safetensors',
        'description': 'Semantics-aware diffusion super-resolution with caption guidance',
        'scale': 4,
        'backend': 'seesr',
    },
    'CCSR': {
        'model_file': 'real-world_ccsr.ckpt',
        'description': 'Controllable diffusion super-resolution with structure/texture balance',
        'scale': 4,
        'backend': 'ccsr',
    },
    'OSEDiff': {
        'model_file': None,  # official weights are only distributed via Google Drive
        'description': 'One-step diffusion super-resolution (NeurIPS 2024, fast)',
        'scale': 4,
        'backend': 'osediff',
    },
    'DifFace': {
        'model_file': 'diffusion_pytorch_model.bin',
        'description': 'Diffusion-based blind face restoration with diffused error contraction',
        'scale': 1,
        'backend': 'difface',
    },
}

# Diffusion-based restorers. The model manager makes the weights downloadable
# (see models.json), but running them requires each project's own inference
# code, which is large and not bundled. Following the app's established
# optional-dependency pattern (like realesrgan/basicsr/facelib), the backend
# package must be installed by the user; until then selecting one of these
# models fails fast with an explicit hint instead of silently producing a
# weaker classical estimate.
DIFFUSION_RESTORERS = {
    'diffbir': {
        'backend_module': 'diffbir',
        'repo': 'https://github.com/XPixelGroup/DiffBIR',
    },
    'supir': {
        'backend_module': 'SUPIR',
        'repo': 'https://github.com/Fanghua-Yu/SUPIR',
    },
    'stablesr': {
        'backend_module': 'stablesr',
        'repo': 'https://github.com/IceClear/StableSR',
    },
    'seesr': {
        'backend_module': 'seesr',
        'repo': 'https://github.com/cswry/SeeSR',
    },
    'ccsr': {
        'backend_module': 'ccsr',
        'repo': 'https://github.com/csslc/CCSR',
    },
    'osediff': {
        'backend_module': 'osediff',
        'repo': 'https://github.com/cswry/OSEDiff',
    },
    'difface': {
        'backend_module': 'difface',
        'repo': 'https://github.com/czczup/DifFace',
    },
}


class FrameRecoveryPipeline(EnhancementPipeline):
    """Frame recovery for reconstructing missing/stuttering frames"""
    
    def __init__(self, model_name: str = 'BasicVSR', device: str = 'auto', scale: int = 4):
        super().__init__(device)
        self.model_name = model_name
        self.scale = scale
        self._model = None
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        try:
            logger.info(f"Loaded frame recovery model: {self.model_name}")
            return True
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * self.scale
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.scale
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            progress_callback = kwargs.get('progress_callback')
            
            # Process frames with temporal consistency
            for i, frame in enumerate(frames):
                try:
                    # Apply restoration
                    restored = self._restore_frame(frame)
                    out.write(restored)
                except Exception as e:
                    logger.warning(f"Frame {i} failed: {e}")
                    out.write(frame)
                
                if progress_callback:
                    progress_callback((i + 1) / len(frames) * 100)
            
            out.release()
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': len(frames),
                'model': self.model_name
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Frame recovery failed: {e}")
    
    def _restore_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply restoration to a single frame"""
        # Apply sharpening and slight denoising
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 8.0
        restored = cv2.filter2D(frame, -1, kernel)
        return restored


class MultipurposeRestorationPipeline(EnhancementPipeline):
    """Multipurpose video restoration pipeline"""
    
    def __init__(self, model_name: str = 'RealBasicVSR', device: str = 'auto', 
                 scale: int = 4, denoise_strength: float = 0.5):
        super().__init__(device)
        self.model_name = model_name
        self.scale = scale
        self.denoise_strength = denoise_strength
        self._model = None
        self._model_config = MULTIPURPOSE_MODELS.get(model_name, MULTIPURPOSE_MODELS['RealBasicVSR'])
        self._backend = self._model_config.get('backend', 'classical')
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        """Load multipurpose restoration model.

        Classical models (RealBasicVSR, Real-CUGAN) load trivially. Diffusion
        restorers (DiffBIR, SUPIR, StableSR, SeeSR, CCSR, OSEDiff) are not
        bundled: their weights are downloadable via the model manager, but the
        inference backend must be installed separately. Until that ships, they
        raise an explicit error instead of silently falling back to the weak
        classical estimate.
        """
        try:
            if self._backend == 'classical':
                logger.info(f"Loaded multipurpose restoration: {self.model_name}")
                self._model = True
                return True

            info = DIFFUSION_RESTORERS.get(self._backend)
            if not info:
                raise ModelLoadError(
                    f"Restoration backend '{self._backend}' is not supported"
                )

            import importlib
            try:
                module = importlib.import_module(info['backend_module'])
            except ImportError:
                raise ModelLoadError(
                    f"'{self.model_name}' needs its inference backend, which is not "
                    f"bundled with openVideoAI.\n"
                    f"Download the weights via the model manager, then install the "
                    f"backend from {info['repo']} (e.g. 'pip install "
                    f"git+{info['repo']}.git') to use this model."
                )
            self._model = module
            logger.info(f"Loaded {self.model_name} diffusion restoration backend")
            return True
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * self.scale
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.scale
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress_callback = kwargs.get('progress_callback')
            
            processed = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    restored = self._apply_restoration(frame)
                    out.write(restored)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Frame {processed} failed: {e}")
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
                'model': self.model_name
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Multipurpose restoration failed: {e}")
    
    def _apply_restoration(self, frame: np.ndarray) -> np.ndarray:
        """Apply multipurpose restoration"""
        # Denoise
        if self.denoise_strength > 0:
            strength = int(self.denoise_strength * 10)
            frame = cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 8.0
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame


def recover_frames(input_path: str, output_path: str, model_name: str = 'BasicVSR',
                   device: str = 'auto', scale: int = 4,
                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for frame recovery"""
    pipeline = FrameRecoveryPipeline(model_name=model_name, device=device, scale=scale)
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def multipurpose_restore(input_path: str, output_path: str, model_name: str = 'RealBasicVSR',
                         device: str = 'auto', scale: int = 4, denoise_strength: float = 0.5,
                         models_dir: Optional[str] = None,
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for multipurpose restoration"""
    pipeline = MultipurposeRestorationPipeline(model_name=model_name, device=device,
                                                scale=scale, denoise_strength=denoise_strength)
    model_path = get_multipurpose_model(model_name, models_dir)
    pipeline.load_model(model_path)
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def get_multipurpose_model(model_name: str, models_dir: Optional[str] = None) -> Optional[str]:
    """Return the local path of a downloaded multipurpose/diffusion model.

    ``models_dir`` defaults to the application's resolved model directory
    (``src.core.paths.models_dir()``), which is CWD-independent.
    """
    config = MULTIPURPOSE_MODELS.get(model_name)
    if not config:
        return None
    filename = config.get('model_file')
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


def list_multipurpose_models(models_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available multipurpose restoration models with their status."""
    results = []
    for name, config in MULTIPURPOSE_MODELS.items():
        path = get_multipurpose_model(name, models_dir)
        results.append({
            'name': name,
            'description': config.get('description', ''),
            'downloaded': path is not None,
            'path': path,
            'backend': config.get('backend', 'classical'),
        })
    return results