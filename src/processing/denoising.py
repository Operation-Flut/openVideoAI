"""
Denoise processing using SCUNet and other denoising models
"""
import os
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

# Denoise Model Configurations
DENOISE_MODELS = {
    'NAFNet-SIDD': {
        'model_file': 'NAFNet-SIDD-width64.pth',
        'description': 'NAFNet trained on the SIDD dataset for image denoising',
        'tile_size': 512,
    },
    'NAFNet': {
        'model_file': 'NAFNet-SIDD-width64.pth',
        'description': 'Legacy name for NAFNet-SIDD image denoising',
        'tile_size': 512,
    },
    'SCUNet': {
        'model_file': 'scunet_color_real_psnr.pth',
        'description': 'Swin Transformer UNet for image denoising',
        'tile_size': 512,
    },
    'Uformer': {
        'model_file': 'uformer.pth',
        'description': 'Transformer-based image restoration',
        'tile_size': 512,
    },
    'MIRNet': {
        'model_file': 'mirnet.pth',
        'description': 'Multi-scale Interactive Residual Network',
        'tile_size': 256,
    },
    'DnCNN': {
        'model_file': 'dncnn.pth',
        'description': 'Deep Convolutional Neural Network for denoising',
        'tile_size': 512,
    }
}

# NAFNet-SIDD-width64.pth was trained with a different block layout than the
# GoPro/REDS width-64 models: encoders [2, 2, 4, 8], 12 middle blocks and
# decoders [2, 2, 2, 2]. Loading those weights with the GoPro profile
# ([1, 1, 1, 28] / 1 / [1, 1, 1, 1]) cannot be strict-loaded, so the Denoise
# pipeline overrides the SharpenPipeline architecture config with this one.
_NAFNET_SIDD_ARCH_CONFIG: Dict[str, Any] = {
    'width': 64,
    'middle_blk_num': 12,
    'enc_blk_nums': [2, 2, 4, 8],
    'dec_blk_nums': [2, 2, 2, 2],
}


class DenoisePipeline(EnhancementPipeline):
    """Denoise enhancement pipeline"""
    
    def __init__(self, model_name: str = 'SCUNet', device: str = 'auto',
                 strength: float = 0.5, tile_size: int = 512):
        super().__init__(device)
        self.model_name = model_name
        self.strength = strength
        self.tile_size = tile_size
        self._model = None
        self._nafnet_pipeline = None
        self._model_config = DENOISE_MODELS.get(model_name, DENOISE_MODELS['SCUNet'])
    
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load denoising model"""
        try:
            import torch

            if self.model_name in ('NAFNet', 'NAFNet-SIDD'):
                if not model_path or not Path(model_path).exists():
                    raise ModelLoadError(f"Model file not found: {model_path}")

                from .sharpening import SharpenPipeline

                # Both the modern 'NAFNet-SIDD' and the legacy 'NAFNet' name
                # ship the SIDD checkpoint, so they share the SIDD block layout.
                self._nafnet_pipeline = SharpenPipeline(
                    model_name='NAFNet',
                    device=self.device,
                    strength=self.strength,
                    tile_size=self.tile_size,
                    arch_config=_NAFNET_SIDD_ARCH_CONFIG,
                )
                self._nafnet_pipeline.load_model(model_path)
                logger.info(f"Loaded {self.model_name} denoising model on {self.device}")
                return True

            from .archs.scunet_arch import SCUNet

            if self.model_name == 'SCUNet':
                self._model = SCUNet(
                    in_nc=3,
                    config=[4, 4, 4, 4, 4, 4, 4],
                    dim=64,
                    drop_path_rate=0.0,
                    input_resolution=512,
                )
            else:
                raise ModelLoadError(
                    f"Architecture for '{self.model_name}' is not available in this build"
                )

            if not model_path or not Path(model_path).exists():
                raise ModelLoadError(f"Model file not found: {model_path}")

            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            state = checkpoint.get('params', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self._model.load_state_dict(state, strict=True)
            
            self._model = self._model.to(self.device)
            self._model.eval()
            
            logger.info(f"Loaded {self.model_name} denoising model on {self.device}")
            return True
            
        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process image or video for denoising"""
        if self.model_name in ('NAFNet', 'NAFNet-SIDD'):
            if self._nafnet_pipeline is None:
                raise ModelLoadError("NAFNet denoising model has not been loaded")
            result = self._nafnet_pipeline.process(input_path, output_path, **kwargs)
            result['model'] = self.model_name
            return result

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
        """Process single image for denoising"""
        import torch
        
        progress_callback = kwargs.get('progress_callback')
        
        try:
            if progress_callback:
                progress_callback(10)
            
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if progress_callback:
                progress_callback(30)
            
            if self._model:
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self._model(img_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    output = output.squeeze(0).permute(1, 2, 0).cpu().float()
                    output = (output * 255).clamp(0, 255).numpy().astype(np.uint8)
                
                # Blend with original based on strength
                if self.strength < 1.0:
                    output = cv2.addWeighted(img_rgb, 1 - self.strength, output, self.strength, 0)
                
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            else:
                output_bgr = self._classical_denoise(img)
            
            if progress_callback:
                progress_callback(80)
            
            if not cv2.imwrite(output_path, output_bgr):
                raise InferenceError(f"Failed to write output image: {output_path}")
            
            if progress_callback:
                progress_callback(100)
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'input_shape': img.shape[:2],
                'output_shape': output_bgr.shape[:2],
                'model': self.model_name,
                'strength': self.strength
            }
            
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Denoise failed: {e}")
    
    def _classical_denoise(self, img: np.ndarray) -> np.ndarray:
        """Classical denoising fallback"""
        # Non-local means denoising
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
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
            progress_callback = kwargs.get('progress_callback')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    temp_input = f"_temp_denoise_frame_{processed}.jpg"
                    cv2.imwrite(temp_input, frame)
                    result = self._process_image(temp_input, output_path, **{})
                    
                    if result.get('success'):
                        restored = cv2.imread(output_path)
                        out.write(restored)
                        processed += 1
                        try:
                            Path(temp_input).unlink()
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
            raise InferenceError(f"Denoise video failed: {e}")


def get_denoise_model(model_name: str, models_dir: str = "models") -> Optional[str]:
    """Get path to denoise model file"""
    model_config = DENOISE_MODELS.get(model_name)
    if not model_config:
        return None
    
    filename = model_config.get('model_file')
    if not filename:
        return None
    
    path = Path(models_dir) / filename
    if path.exists():
        return str(path)
    return None


def denoise_image(input_path: str, output_path: str, model_name: str = 'SCUNet',
                  device: str = 'auto', models_dir: str = "models",
                  progress_callback: Optional[callable] = None,
                  strength: float = 0.5) -> Dict[str, Any]:
    """Convenience function for denoising"""
    model_path = get_denoise_model(model_name, models_dir)
    
    pipeline = DenoisePipeline(model_name=model_name, device=device, strength=strength)
    pipeline.load_model(model_path)
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def list_denoise_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """List available denoise models"""
    results = []
    for name, config in DENOISE_MODELS.items():
        path = get_denoise_model(name, models_dir)
        results.append({
            'name': name,
            'description': config['description'],
            'downloaded': path is not None,
            'path': path
        })
    return results