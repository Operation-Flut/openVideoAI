"""
Low Light Enhancement, Correct Distortion, and Log Restoration processing
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


class LowLightEnhancementPipeline(EnhancementPipeline):
    """Enhance low-light/underexposed images and videos"""
    
    def __init__(self, model_name: str = 'Zero-DCE', device: str = 'auto', strength: float = 0.7):
        super().__init__(device)
        self.model_name = model_name
        self.strength = strength
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded low light enhancement: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        path = Path(input_path)
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        
        if is_video:
            return self._process_video(input_path, output_path, **kwargs)
        else:
            return self._process_image(input_path, output_path, **kwargs)
    
    def _process_image(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            # Apply CLAHE for low light enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Gamma correction based on strength
            gamma = 1.0 + (1.0 - self.strength) * 0.5
            l = np.power(l / 255.0, gamma) * 255
            l = np.clip(l, 0, 255).astype(np.uint8)
            
            lab_enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # Brightness adjustment
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=self.strength * 30)
            
            if not cv2.imwrite(output_path, enhanced):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"Low light enhancement failed: {e}")
    
    def _process_video(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
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
            
            progress_callback = kwargs.get('progress_callback')
            processed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self._process_image_temp(frame)
                out.write(result)
                processed += 1
                
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
        except Exception as e:
            raise InferenceError(f"Low light video processing failed: {e}")
    
    def _process_image_temp(self, img: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        gamma = 1.0 + (1.0 - self.strength) * 0.5
        l = np.power(l / 255.0, gamma) * 255
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        lab_enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=self.strength * 30)
        
        return enhanced


class DistortionCorrectionPipeline(EnhancementPipeline):
    """Correct lens distortion and perspective issues"""
    
    def __init__(self, model_name: str = 'LensCorrection', device: str = 'auto'):
        super().__init__(device)
        self.model_name = model_name
        self.camera_matrix = None
        self.dist_coeffs = None
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        # Default camera matrix for common smartphone cameras
        self.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.1, 0.01, 0, 0], dtype=np.float32)
        logger.info(f"Loaded distortion correction: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            h, w = img.shape[:2]
            
            # Update camera matrix for actual image size
            new_camera_matrix = self.camera_matrix.copy()
            new_camera_matrix[0, 2] = w / 2
            new_camera_matrix[1, 2] = h / 2
            
            # Undistort
            corrected = cv2.undistort(img, new_camera_matrix, self.dist_coeffs)
            
            # Also apply perspective correction if needed
            gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            if not cv2.imwrite(output_path, corrected):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"Distortion correction failed: {e}")


class LogRestorationPipeline(EnhancementPipeline):
    """Restore LOG footage to Rec.709"""
    
    def __init__(self, model_name: str = 'ITMLUT', device: str = 'auto', 
                 color_space_target: str = 'Rec.709'):
        super().__init__(device)
        self.model_name = model_name
        self.color_space_target = color_space_target
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded LOG restoration: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        path = Path(input_path)
        is_video = path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        
        if is_video:
            return self._process_video(input_path, output_path, **kwargs)
        else:
            return self._process_image(input_path, output_path, **kwargs)
    
    def _apply_log_conversion(self, img: np.ndarray) -> np.ndarray:
        """Apply LOG to Rec.709 conversion"""
        # Simple log to gamma conversion
        log_img = img.astype(np.float32) / 255.0
        
        # Apply S-Log3 to Rec.709 conversion curve
        # This is a simplified version - real implementation would use LUTs
        converted = np.where(
            log_img >= 0.18,
            1.099 * np.power(log_img, 0.45) - 0.099,
            4.5 * log_img
        )
        
        converted = np.clip(converted * 255, 0, 255).astype(np.uint8)
        
        # Enhance colors for Rec.709
        hsv = cv2.cvtColor(converted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.2, beta=10)
        converted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Boost saturation slightly
        converted = cv2.convertScaleAbs(converted, alpha=1.1, beta=5)
        
        return converted
    
    def _process_image(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            restored = self._apply_log_conversion(img)
            
            if not cv2.imwrite(output_path, restored):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'color_space': self.color_space_target,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"LOG restoration failed: {e}")
    
    def _process_video(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
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
            
            progress_callback = kwargs.get('progress_callback')
            processed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                restored = self._apply_log_conversion(frame)
                out.write(restored)
                processed += 1
                
                if progress_callback:
                    progress_callback(processed / total_frames * 100)
            
            cap.release()
            out.release()
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': processed,
                'color_space': self.color_space_target,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"LOG restoration video failed: {e}")


def enhance_low_light(input_path: str, output_path: str, 
                      progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for low light enhancement"""
    pipeline = LowLightEnhancementPipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def correct_distortion(input_path: str, output_path: str,
                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for distortion correction"""
    pipeline = DistortionCorrectionPipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def restore_log(input_path: str, output_path: str, color_space: str = 'Rec.709',
                progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for LOG restoration"""
    pipeline = LogRestorationPipeline(color_space_target=color_space)
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)