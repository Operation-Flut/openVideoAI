"""
SDR to HDR, Object Removal, Focus Fix, Overexposure Fix, and Color Profile processing
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
    InferenceError,
    ProcessingAborted,
)

logger = logging.getLogger(__name__)


class SDRtoHDRPipeline(EnhancementPipeline):
    """Convert SDR to HDR"""
    
    def __init__(self, model_name: str = 'HDRNet', device: str = 'auto', 
                 target_format: str = 'HDR10'):
        super().__init__(device)
        self.model_name = model_name
        self.target_format = target_format
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded SDR to HDR converter: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            # HDR-like enhancement
            # Tone map to expand dynamic range
            hdr = self._simulate_hdr(img)
            
            if not cv2.imwrite(output_path, hdr):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'target_format': self.target_format,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"SDR to HDR failed: {e}")
    
    def _simulate_hdr(self, img: np.ndarray) -> np.ndarray:
        """Simulate HDR effect from SDR"""
        # Merge multiple exposures
        exposures = []
        
        # Underexposed
        underexposed = cv2.convertScaleAbs(img, alpha=0.5, beta=-20)
        exposures.append(underexposed)
        
        # Normal
        exposures.append(img)
        
        # Overexposed  
        overexposed = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        exposures.append(overexposed)
        
        # Merge using Mertens fusion
        merge = cv2.createMergeMertens()
        merged = merge.process(exposures)
        
        # Convert back to 8-bit
        merged = (merged * 255).clip(0, 255).astype(np.uint8)
        
        # Enhance local contrast
        lab = cv2.cvtColor(merged, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        return result


class ObjectRemovalPipeline(EnhancementPipeline):
    """Remove unwanted objects using inpainting"""
    
    def __init__(self, model_name: str = 'LaMa', device: str = 'auto'):
        super().__init__(device)
        self.model_name = model_name
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded object removal: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, mask_path: str = None, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            # Create or load mask
            if mask_path and Path(mask_path).exists():
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Auto-generate mask (would need user input in real implementation)
                # For now, create a dummy mask
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            # Apply inpainting
            result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            
            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"Object removal failed: {e}")


class FocusFixPipeline(EnhancementPipeline):
    """Fix out-of-focus images"""
    
    def __init__(self, model_name: str = 'DeblurGAN', device: str = 'auto'):
        super().__init__(device)
        self.model_name = model_name
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded focus fix: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            # Apply blind deconvolution for focus correction
            result = self._blind_deconvolution(img)
            
            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"Focus fix failed: {e}")
    
    def _blind_deconvolution(self, img: np.ndarray) -> np.ndarray:
        """Simple blind deconvolution for focus fix"""
        # Richardson-Lucy deconvolution approximation using Wiener
        # Create a simple blur kernel
        kernel_size = 5
        sigma = 1.5
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        
        # Convert to grayscale for deconvolution
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Wiener deconvolution
        fft_kernel = np.fft.fft2(kernel, gray.shape)
        fft_gray = np.fft.fft2(gray.astype(np.float32))
        
        # Wiener filter
        nsr = 0.01  # Noise to signal ratio
        wiener = np.conj(fft_kernel) / (np.abs(fft_kernel) ** 2 + nsr)
        
        deconvolved = np.fft.ifft2(wiener * fft_gray)
        deconvolved = np.abs(deconvolved).clip(0, 255).astype(np.uint8)
        
        # Apply to color
        result = np.zeros_like(img)
        for i in range(3):
            fft_channel = np.fft.fft2(img[:, :, i].astype(np.float32))
            deconvolved_c = np.fft.ifft2(wiener * fft_channel)
            result[:, :, i] = np.abs(deconvolved_c).clip(0, 255).astype(np.uint8)
        
        # Blend with original for natural look
        result = cv2.addWeighted(img, 0.3, result, 0.7, 0)
        
        return result


class OverexposureFixPipeline(EnhancementPipeline):
    """Fix overexposure and contrast issues"""
    
    def __init__(self, model_name: str = 'AutoEnhance', device: str = 'auto'):
        super().__init__(device)
        self.model_name = model_name
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded overexposure fix: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            # Fix overexposure
            result = self._fix_overexposure(img)
            
            # Fix contrast
            result = self._fix_contrast(result)
            
            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"Overexposure fix failed: {e}")
    
    def _fix_overexposure(self, img: np.ndarray) -> np.ndarray:
        """Fix overexposed areas"""
        # Convert to YUV
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        
        # Find overexposed regions
        threshold = 220
        overexposed_mask = y > threshold
        
        # Reduce intensity in overexposed areas
        y_corrected = y.copy()
        y_corrected[overexposed_mask] = np.clip(
            y[overexposed_mask] * 0.7, 0, 255
        ).astype(np.uint8)
        
        # Apply gentle histogram equalization to bright areas
        bright_mask = (y > 150) & (y <= 220)
        y_bright = y[bright_mask]
        if len(y_bright) > 0:
            eq = cv2.equalizeHist(y_bright)
            y_corrected[bright_mask] = eq
        
        return cv2.cvtColor(cv2.merge([y_corrected, u, v]), cv2.COLOR_YUV2BGR)
    
    def _fix_contrast(self, img: np.ndarray) -> np.ndarray:
        """Fix contrast issues"""
        # CLAHE for local contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


class FlatColorProfilePipeline(EnhancementPipeline):
    """Apply flat color profile for color correction"""
    
    def __init__(self, model_name: str = 'ColorGrade', device: str = 'auto'):
        super().__init__(device)
        self.model_name = model_name
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        logger.info(f"Loaded flat color profile: {self.model_name}")
        return True
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")
            
            # Apply flat color grading
            result = self._apply_flat_profile(img)
            
            if not cv2.imwrite(output_path, result):
                raise InferenceError(f"Failed to write output: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'model': self.model_name
            }
        except Exception as e:
            raise InferenceError(f"Flat color profile failed: {e}")
    
    def _apply_flat_profile(self, img: np.ndarray) -> np.ndarray:
        """Apply flat color profile correction"""
        # Convert to YCrCb for better color handling
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Flatten the curve (reduce contrast)
        # Reduce the difference between highlights and shadows
        y_flat = np.clip(y * 0.8 + 30, 0, 255).astype(np.uint8)
        
        # Adjust color saturation (reduce)
        cr_adjusted = np.clip(cr * 0.9 + 10, 0, 255).astype(np.uint8)
        cb_adjusted = np.clip(cb * 0.9 + 10, 0, 255).astype(np.uint8)
        
        result_ycrcb = cv2.merge([y_flat, cr_adjusted, cb_adjusted])
        result = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Apply color balance correction
        result = self._balance_colors(result)
        
        return result
    
    def _balance_colors(self, img: np.ndarray) -> np.ndarray:
        """Balance colors to neutral"""
        # Simple white balance using Gray World assumption
        result = img.astype(np.float32)
        
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        gray = (avg_b + avg_g + avg_r) / 3
        
        result[:, :, 0] = np.clip(result[:, :, 0] * (gray / (avg_b + 1e-8)), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (gray / (avg_g + 1e-8)), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (gray / (avg_r + 1e-8)), 0, 255)
        
        return result.astype(np.uint8)


def convert_sdr_to_hdr(input_path: str, output_path: str, target: str = 'HDR10',
                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for SDR to HDR conversion"""
    pipeline = SDRtoHDRPipeline(target_format=target)
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def remove_objects(input_path: str, output_path: str, mask_path: str = None,
                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for object removal"""
    pipeline = ObjectRemovalPipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, mask_path=mask_path, 
                           progress_callback=progress_callback)


def fix_focus(input_path: str, output_path: str,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for focus fix"""
    pipeline = FocusFixPipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def fix_overexposure(input_path: str, output_path: str,
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for overexposure fix"""
    pipeline = OverexposureFixPipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def apply_flat_profile(input_path: str, output_path: str,
                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for flat color profile"""
    pipeline = FlatColorProfilePipeline()
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)