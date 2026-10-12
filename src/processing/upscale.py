"""
Upscale processing using Real-ESRGAN and other upscale models
Supports all upscale models from the catalog
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
from builtins import UnboundLocalError

from ..distutils_compat import install_distutils_compat

install_distutils_compat()

from .base import (
    EnhancementPipeline,
    ModelLoadError,
    InferenceError,
    UnsupportedFormatError,
    ProcessingAborted,
)

logger = logging.getLogger(__name__)

# Extended model configurations - now including all catalog upscale models
UPSCALE_MODELS = {
    'Real-ESRGAN': {
        'class': 'RealESRGANer',
        'scale': 4,
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'scale': 4
        },
        'tile_size': 400,
        'tile_pad': 10,
        'pre_pad': 0,
        'half_precision': True,
        'model_file': 'RealESRGAN_x4plus.pth',
        'description': 'General purpose image/video upscaling'
    },
    'RealESRGAN_x4plus_anime_6B': {
        'class': 'RealESRGANer',
        'scale': 4,
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 6,
            'scale': 4
        },
        'tile_size': 400,
        'tile_pad': 10,
        'pre_pad': 0,
        'half_precision': True,
        'model_file': 'RealESRGAN_x4plus_anime_6B.pth',
        'description': 'Anime-style image/video upscaling'
    },
    'RealESRGAN_x2plus': {
        'class': 'RealESRGANer',
        'scale': 2,
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'scale': 2
        },
        'tile_size': 400,
        'tile_pad': 10,
        'pre_pad': 0,
        'half_precision': True,
        'model_file': 'RealESRGAN_x2plus.pth',
        'description': '2x image/video upscaling'
    },
    'Real-CUGAN': {
        'class': 'RealESRGANer',
        'scale': 3,  # CUGAN supports 2x, 3x, 4x
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 16,
            'scale': 3
        },
        'tile_size': 256,
        'tile_pad': 10,
        'pre_pad': 0,
        'half_precision': True,
        'model_file': 'realcugan_nose.pt',  # Placeholder - actual file depends on version
        'description': 'Anime/multipurpose upscaling with denoising'
    },
    'RealBasicVSR': {
        'class': 'RealBasicVSR',
        'scale': 4,
        'model_params': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 10,
            'scale': 4
        },
        'tile_size': 0,  # Video models process sequentially
        'tile_pad': 0,
        'pre_pad': 0,
        'half_precision': True,
        'model_file': 'RealBasicVSR_x4.pth',
        'description': 'Video upscaling with temporal consistency'
    }
}

# Global model cache
_model_cache: Dict[str, Any] = {}


class UpscalePipeline(EnhancementPipeline):
    """Upscale enhancement pipeline using Real-ESRGAN and other models"""
    
    def __init__(self, model_name: str = 'Real-ESRGAN', device: str = 'auto', 
                 scale: int = 4, tile_size: int = 400, fp16: bool = True,
                 jpg_quality: int = 95):
        super().__init__(device)
        self.model_name = model_name
        self.target_scale = scale
        self.tile_size = tile_size
        self.fp16 = fp16
        self.jpg_quality = max(1, min(100, jpg_quality))
        self._upsampler = None
        self._model_config = UPSCALE_MODELS.get(model_name, UPSCALE_MODELS['Real-ESRGAN'])
    
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load upscale model"""
        try:
            import torch
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            config = self._model_config
            
            # Create model architecture
            model = RRDBNet(**config['model_params'])
            
            # Create upsampler
            self._upsampler = RealESRGANer(
                scale=config['scale'],
                model_path=model_path,
                model=model,
                tile=self.tile_size,
                tile_pad=config['tile_pad'],
                pre_pad=config['pre_pad'],
                half=config['half_precision'] and self.fp16 and self.device.startswith(('cuda', 'hip', 'dml', 'privateuseone')),
                device=torch.device(self.device)
            )
            
            self._model_name = self.model_name
            logger.info(f"Loaded {self.model_name} on {self.device}")
            return True
            
        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process image or video"""
        if not self._upsampler:
            raise InferenceError("Model not loaded. Call load_model() first.")
        
        path = Path(input_path)
        if not path.exists():
            raise InferenceError(f"Input file not found: {input_path}")
        
        # Determine if video or image
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        is_video = path.suffix.lower() in video_extensions
        
        if is_video:
            return self._process_video(input_path, output_path, **kwargs)
        else:
            return self._process_image(input_path, output_path, **kwargs)
    
    def _process_image(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process single image with optional progress callback for tiling"""
        import torch
        
        progress_callback = kwargs.get('progress_callback')
        
        try:
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Check if image is large enough to need tiling
            if max_dim > self.tile_size and progress_callback:
                output = self._process_image_tiled(img, progress_callback)
            else:
                if progress_callback:
                    progress_callback(10)  # Starting
                with torch.no_grad():
                    output, _ = self._upsampler.enhance(img, outscale=self.target_scale)
                if progress_callback:
                    progress_callback(90)
            
            # Convert back to BGR for saving
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Save
            write_params = []
            if Path(output_path).suffix.lower() in {'.jpg', '.jpeg'}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
            if not cv2.imwrite(output_path, output, write_params):
                raise InferenceError(f"Failed to write output image: {output_path}")
            
            if progress_callback:
                progress_callback(100)
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'input_shape': img.shape[:2],
                'output_shape': output.shape[:2],
                'scale_factor': self.target_scale,
                'model': self.model_name
            }
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Image upscale failed: {e}")

    def _process_image_tiled(self, img: np.ndarray, progress_callback: callable) -> np.ndarray:
        """Process large image using custom tiling with progress reporting"""
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
                        # Handle Real-ESRGAN's internal UnboundLocalError and VRAM errors
                        if isinstance(e, UnboundLocalError) and 'output_tile' in str(e):
                            # This is a known bug in Real-ESRGAN's tile processing
                            # Try to recover by skipping this tile or reducing tile size
                            logger.warning(f"Tile processing error (output_tile), skipping tile at ({tx},{ty}): {e}")
                            continue
                        elif "could not allocate tensor" in str(e).lower():
                            # VRAM error - try to continue with smaller tile if possible
                            logger.warning(f"Tile VRAM error, skipping tile at ({tx},{ty}): {e}")
                            continue
                        else:
                            # Re-raise other errors
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
                    if progress_callback:
                        progress = 10 + int((tile_count / total_tiles) * 80)
                        progress_callback(progress)
        
        return output

    def _process_video(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process video frame by frame"""
        import torch
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out_width = width * self.target_scale
            out_height = height * self.target_scale
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")
            
            processed = 0
            failed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    with torch.no_grad():
                        upscaled, _ = self._upsampler.enhance(frame_rgb, outscale=self.target_scale)
                    
                    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                    
                    out.write(upscaled_bgr)
                    processed += 1
                    
                except Exception as e:
                    logger.warning(f"Frame {processed} failed: {e}")
                    failed += 1
                    fallback = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
                    out.write(fallback)
                
                if 'progress_callback' in kwargs:
                    kwargs['progress_callback'](processed / total_frames * 100)
            
            cap.release()
            out.release()
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': processed,
                'frames_failed': failed,
                'total_frames': total_frames,
                'input_resolution': f"{width}x{height}",
                'output_resolution': f"{out_width}x{out_height}",
                'scale_factor': self.target_scale,
                'fps': fps,
                'model': self.model_name
            }
            
        except ProcessingAborted:
            try:
                cap.release()
            except Exception:
                pass
            try:
                out.release()
            except Exception:
                pass
            raise
        except Exception as e:
            try:
                cap.release()
            except Exception:
                pass
            try:
                out.release()
            except Exception:
                pass
            raise InferenceError(f"Video upscale failed: {e}")


def get_upscale_model(model_name: str, models_dir: Optional[str] = None) -> Optional[str]:
    """Get path to upscale model file.

    ``models_dir`` defaults to the application's resolved model directory
    (``src.core.paths.models_dir()``), which is CWD-independent; a plain
    relative ``"models"`` value is still honoured for backwards compatibility.
    """
    model_config = UPSCALE_MODELS.get(model_name)
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


def get_all_upscale_models_from_catalog(models_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all upscale models that have model files available"""
    import json
    from pathlib import Path as P

    if models_dir is None:
        try:
            from ..core import paths as _paths
            models_dir = str(_paths.models_dir())
        except Exception:
            models_dir = "models"
    
    results = []
    
    # First, check our known models
    for name, config in UPSCALE_MODELS.items():
        path = get_upscale_model(name, models_dir)
        results.append({
            'name': name,
            'scale': config['scale'],
            'description': config.get('description', ''),
            'downloaded': path is not None,
            'path': path
        })
    
    # Also load from models.json catalog for any additional upscale models
    catalog_path = P(models_dir).parent / 'models.json'
    if catalog_path.exists():
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for model in data.get('models', []):
                features = model.get('features', [])
                has_upscale = any(f.get('name') == 'Upscale' for f in features)
                if has_upscale:
                    # Check if we already have this model
                    existing = next((r for r in results if r['name'] == model['name']), None)
                    if not existing:
                        results.append({
                            'name': model['name'],
                            'scale': 4,  # Default
                            'description': model.get('descriptionShort', ''),
                            'downloaded': False,
                            'path': None
                        })
        except Exception as e:
            logger.warning(f"Could not load catalog: {e}")
    
    return results


def upscale_image(input_path: str, output_path: str, model_name: str = 'Real-ESRGAN',
                  scale: int = 4, device: str = 'auto', models_dir: Optional[str] = None,
                  progress_callback: Optional[callable] = None, tile_size: int = 400,
                  fp16: bool = True, jpg_quality: int = 95) -> Dict[str, Any]:
    """Convenience function to upscale a single image"""
    model_path = get_upscale_model(model_name, models_dir)
    if not model_path:
        raise InferenceError(f"Model file not found for {model_name}. Download it first.")
    
    pipeline = UpscalePipeline(
        model_name=model_name, device=device, scale=scale, tile_size=tile_size,
        fp16=fp16, jpg_quality=jpg_quality
    )
    pipeline.load_model(model_path)
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def upscale_video(input_path: str, output_path: str, model_name: str = 'Real-ESRGAN',
                  scale: int = 4, device: str = 'auto', models_dir: Optional[str] = None,
                  progress_callback: Optional[callable] = None, tile_size: int = 400,
                  fp16: bool = True, jpg_quality: int = 95) -> Dict[str, Any]:
    """Convenience function to upscale a video"""
    model_path = get_upscale_model(model_name, models_dir)
    if not model_path:
        raise InferenceError(f"Model file not found for {model_name}. Download it first.")
    
    pipeline = UpscalePipeline(
        model_name=model_name, device=device, scale=scale, tile_size=tile_size,
        fp16=fp16, jpg_quality=jpg_quality
    )
    pipeline.load_model(model_path)
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def list_available_upscale_models(models_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available upscale models with their status"""
    return get_all_upscale_models_from_catalog(models_dir)