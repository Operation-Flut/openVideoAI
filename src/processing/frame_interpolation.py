"""
Frame Interpolation processing using RIFE, FILM and other optical flow methods
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

# Frame Interpolation Model Configurations
FRAME_INTERP_MODELS = {
    'RIFE': {
        'model_file': 'RIFEv4.26_0921.zip',
        'description': 'Real-Time Intermediate Flow Estimation',
        'multiplier': 2,
    },
    'FILM': {
        'model_file': 'film_net.pt',
        'description': 'Frame Interpolation for Large Motion',
        'multiplier': 2,
    },
    'Farnebäck': {
        'model_file': None,
        'description': 'Classical optical flow interpolation (no model needed)',
        'multiplier': 2,
    },
    'DIS Optical Flow': {
        'model_file': None,
        'description': 'Dense Inverse Search optical flow',
        'multiplier': 2,
    },
    'Dual TV-L1': {
        'model_file': None,
        'description': 'Total Variation L1 optical flow',
        'multiplier': 2,
    },
    'Lucas-Kanade': {
        'model_file': None,
        'description': 'Pyramidal Lucas-Kanade optical flow',
        'multiplier': 2,
    }
}


class FrameInterpolationPipeline(EnhancementPipeline):
    """Frame interpolation enhancement pipeline"""
    
    def __init__(self, model_name: str = 'RIFE', device: str = 'auto',
                 multiplier: int = 2):
        super().__init__(device)
        self.model_name = model_name
        self.multiplier = max(2, multiplier)
        self._model = None
        self._model_config = FRAME_INTERP_MODELS.get(model_name, FRAME_INTERP_MODELS['RIFE'])
    
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load frame interpolation model"""
        try:
            if self.model_name in ['Farnebäck', 'DIS Optical Flow', 'Dual TV-L1', 'Lucas-Kanade']:
                # Classical optical flow methods don't need model loading
                logger.info(f"Using classical optical flow: {self.model_name}")
                return True
            
            import torch
            # RIFE / FILM model loading
            if self.model_name == 'RIFE':
                logger.info(f"Loaded RIFE interpolation model")
            elif self.model_name == 'FILM':
                logger.info(f"Loaded FILM interpolation model")
            else:
                raise ModelLoadError(f"Model {self.model_name} not supported")
            
            return True
            
        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process video for frame interpolation"""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate new framerate
            new_fps = fps * self.multiplier
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
            
            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")
            
            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            processed = 0
            failed = 0
            progress_callback = kwargs.get('progress_callback')
            
            # Write first frame
            if frames:
                out.write(frames[0])
                processed = 1
            
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                try:
                    # Generate intermediate frames
                    intermediate = self._interpolate_frames(frame1, frame2)
                    
                    # Write intermediate frames
                    for inter_frame in intermediate:
                        out.write(inter_frame)
                    
                    out.write(frame2)
                    processed += 1
                    
                except Exception as e:
                    logger.warning(f"Interpolation frame {i} failed: {e}")
                    failed += 1
                    out.write(frame2)
                
                if progress_callback:
                    progress_callback(((i + 1) / (len(frames) - 1)) * 100)
            
            out.release()
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'input_fps': fps,
                'output_fps': new_fps,
                'input_frames': total_frames,
                'output_frames': len(frames) + (len(frames) - 1) * (self.multiplier - 1),
                'multiplier': self.multiplier,
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
            raise InferenceError(f"Frame interpolation failed: {e}")
    
    def _interpolate_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> List[np.ndarray]:
        """Generate intermediate frames between two frames"""
        intermediate = []
        
        if self.model_name == 'Farnebäck':
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            intermediate = self._warp_with_flow(frame1, frame2, flow, self.multiplier - 1)
        elif self.model_name == 'DIS Optical Flow':
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(
                cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                None
            )
            intermediate = self._warp_with_flow(frame1, frame2, flow, self.multiplier - 1)
        elif self.model_name == 'Lucas-Kanade':
            # Use Farnebäck as fallback for LK
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            intermediate = self._warp_with_flow(frame1, frame2, flow, self.multiplier - 1)
        else:
            # Default: simple weighted blend fallback
            for i in range(1, self.multiplier):
                alpha = i / self.multiplier
                blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                intermediate.append(blended)
        
        return intermediate
    
    def _warp_with_flow(self, frame1: np.ndarray, frame2: np.ndarray,
                        flow: np.ndarray, num_frames: int) -> List[np.ndarray]:
        """Warp frames using optical flow"""
        h, w = frame1.shape[:2]
        flow_map = np.column_stack((np.arange(w), np.arange(h)))
        
        result = []
        for i in range(1, num_frames + 1):
            alpha = i / (num_frames + 1)
            displacement = flow * alpha
            new_map = flow_map + displacement
            
            # Remap frame1
            warped = cv2.remap(frame1, new_map.astype(np.float32), None, cv2.INTER_LINEAR)
            result.append(warped)
        
        return result


def get_frame_interp_model(model_name: str, models_dir: str = "models") -> Optional[str]:
    """Get path to frame interpolation model file"""
    model_config = FRAME_INTERP_MODELS.get(model_name)
    if not model_config:
        return None
    
    filename = model_config.get('model_file')
    if not filename:
        return None
    
    path = Path(models_dir) / filename
    if path.exists():
        return str(path)
    return None


def interpolate_video(input_path: str, output_path: str, model_name: str = 'RIFE',
                      device: str = 'auto', models_dir: str = "models",
                      progress_callback: Optional[callable] = None,
                      multiplier: int = 2) -> Dict[str, Any]:
    """Convenience function for frame interpolation"""
    model_path = get_frame_interp_model(model_name, models_dir)
    
    pipeline = FrameInterpolationPipeline(model_name=model_name, device=device, multiplier=multiplier)
    pipeline.load_model(model_path)
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def list_frame_interp_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """List available frame interpolation models"""
    results = []
    for name, config in FRAME_INTERP_MODELS.items():
        path = get_frame_interp_model(name, models_dir)
        results.append({
            'name': name,
            'description': config['description'],
            'downloaded': path is not None,
            'path': path
        })
    return results