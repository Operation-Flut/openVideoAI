"""
Video Reencoding using FFMPEG and Handbrake engines
"""
import os
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import logging
import platform
import json

from ..distutils_compat import install_distutils_compat

install_distutils_compat()

from .base import (
    EnhancementPipeline,
    InferenceError,
    ProcessingAborted,
)

logger = logging.getLogger(__name__)

# Reencoding presets
REENCODING_PRESETS = {
    'h264_fast': {
        'encoder': 'libx264',
        'preset': 'fast',
        'crf': 23,
        'description': 'Fast H.264 encoding, good quality'
    },
    'h264_quality': {
        'encoder': 'libx264',
        'preset': 'slow',
        'crf': 18,
        'description': 'High quality H.264 encoding'
    },
    'h265_fast': {
        'encoder': 'libx265',
        'preset': 'fast',
        'crf': 28,
        'description': 'Fast H.265/HEVC encoding'
    },
    'h265_quality': {
        'encoder': 'libx265',
        'preset': 'slow',
        'crf': 22,
        'description': 'High quality H.265/HEVC encoding'
    },
    'prores': {
        'encoder': 'prores_ks',
        'profile': 'hq',
        'description': 'Apple ProRes high quality'
    },
    'web_optimized': {
        'encoder': 'libx264',
        'preset': 'fast',
        'crf': 23,
        'tune': 'fastdecode',
        'movflags': 'faststart',
        'description': 'Web-optimized H.264'
    }
}

# Handbrake presets
HANDBRAKE_PRESETS = {
    'fast_1080p': {
        'preset': 'Fast 1080p30',
        'description': 'Fast 1080p encoding'
    },
    'hq_1080p': {
        'preset': 'HQ 1080p30 Surround',
        'description': 'High quality 1080p with surround sound'
    },
    'superhq_1080p': {
        'preset': 'Super HQ 1080p30 Surround',
        'description': 'Super high quality 1080p'
    },
    'fast_2160p': {
        'preset': 'Fast 2160p60 4K',
        'description': 'Fast 4K encoding'
    },
    'hq_2160p': {
        'preset': 'HQ 2160p60 4K Surround',
        'description': 'High quality 4K with surround sound'
    }
}


class ReencodingPipeline(EnhancementPipeline):
    """Video reencoding using FFMPEG or Handbrake"""
    
    def __init__(self, engine: str = 'ffmpeg', preset: str = 'h264_fast',
                 device: str = 'auto', custom_params: Dict[str, Any] = None):
        super().__init__(device)
        self.engine = engine
        self.preset = preset
        self.custom_params = custom_params or {}
        self._ffmpeg_path = None
        self._handbrake_path = None
    
    def load_model(self, model_path: str = None, **kwargs) -> bool:
        """Find and verify encoder executables"""
        self._ffmpeg_path = self._find_executable('ffmpeg')
        self._handbrake_path = self._find_executable('HandBrakeCLI')
        
        if self.engine == 'ffmpeg' and not self._ffmpeg_path:
            logger.warning("FFMPEG not found in PATH. Will use OpenCV fallback.")
        elif self.engine == 'handbrake' and not self._handbrake_path:
            logger.warning("HandbrakeCLI not found in PATH. Will use FFMPEG fallback.")
        
        logger.info(f"Loaded reencoding pipeline: {self.engine}")
        return True
    
    def _find_executable(self, name: str) -> Optional[str]:
        """Find executable in PATH"""
        return shutil.which(name)
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Reencode video"""
        path = Path(input_path)
        if not path.exists():
            raise InferenceError(f"Input file not found: {input_path}")
        
        if self.engine == 'handbrake' and self._handbrake_path:
            return self._reencode_handbrake(input_path, output_path, **kwargs)
        elif self._ffmpeg_path:
            return self._reencode_ffmpeg(input_path, output_path, **kwargs)
        else:
            return self._reencode_opencv(input_path, output_path, **kwargs)
    
    def _reencode_ffmpeg(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Reencode using FFMPEG"""
        try:
            preset_config = REENCODING_PRESETS.get(self.preset, REENCODING_PRESETS['h264_fast'])
            
            # Build FFMPEG command
            cmd = [self._ffmpeg_path, '-i', input_path]
            
            # Add encoding parameters
            encoder = preset_config.get('encoder', 'libx264')
            cmd.extend(['-c:v', encoder])
            
            if 'crf' in preset_config:
                cmd.extend(['-crf', str(preset_config['crf'])])
            
            if 'preset' in preset_config:
                cmd.extend(['-preset', preset_config['preset']])
            
            if 'tune' in preset_config:
                cmd.extend(['-tune', preset_config['tune']])
            
            if encoder == 'prores_ks' and 'profile' in preset_config:
                cmd.extend(['-profile:v', preset_config['profile']])
            
            # Audio encoding
            cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
            
            # Output options
            if 'movflags' in preset_config:
                cmd.extend(['-movflags', preset_config['movflags']])
            
            # Overwrite output
            cmd.extend(['-y', output_path])
            
            logger.info(f"Running FFMPEG: {' '.join(cmd)}")
            
            # Run FFMPEG
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            progress_callback = kwargs.get('progress_callback')
            
            # Parse progress from stderr
            for line in process.stderr:
                if progress_callback and 'time=' in line:
                    # Extract time progress
                    try:
                        time_str = line.split('time=')[1].split(' ')[0]
                        progress = self._parse_ffmpeg_progress(time_str, input_path)
                        progress_callback(progress)
                    except:
                        pass
            
            process.wait()
            
            if process.returncode != 0:
                raise InferenceError(f"FFMPEG failed with code {process.returncode}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'engine': 'ffmpeg',
                'preset': self.preset
            }
            
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"FFMPEG reencoding failed: {e}")
    
    def _reencode_handbrake(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Reencode using Handbrake"""
        try:
            preset_config = HANDBRAKE_PRESETS.get(self.preset, HANDBRAKE_PRESETS['fast_1080p'])
            
            cmd = [
                self._handbrake_path,
                '-i', input_path,
                '-o', output_path,
                '--preset', preset_config['preset']
            ]
            
            logger.info(f"Running Handbrake: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.warning(f"Handbrake failed: {process.stderr}")
                # Fallback to FFMPEG
                if self._ffmpeg_path:
                    logger.info("Falling back to FFMPEG")
                    return self._reencode_ffmpeg(input_path, output_path, **kwargs)
                raise InferenceError(f"Handbrake failed: {process.stderr}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'engine': 'handbrake',
                'preset': self.preset
            }
            
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Handbrake reencoding failed: {e}")
    
    def _reencode_opencv(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Fallback reencoding using OpenCV"""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine codec based on preset
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to mp4v
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")
            
            progress_callback = kwargs.get('progress_callback')
            processed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                processed += 1
                
                if progress_callback:
                    progress_callback(processed / total_frames * 100)
            
            cap.release()
            out.release()
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'engine': 'opencv',
                'frames_processed': processed
            }
            
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"OpenCV reencoding failed: {e}")
    
    def _parse_ffmpeg_progress(self, time_str: str, input_path: str) -> float:
        """Parse FFMPEG time string to progress percentage"""
        try:
            # Get total duration
            cap = cv2.VideoCapture(input_path)
            total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Parse current time
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = parts
                current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                return (current_time / total_duration) * 100
        except:
            pass
        return 50.0  # Default progress
    
    def get_video_info(self, input_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe"""
        if not self._ffmpeg_path:
            return {}
        
        try:
            ffprobe = self._ffmpeg_path.replace('ffmpeg', 'ffprobe')
            cmd = [
                ffprobe, '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                input_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except:
            pass
        
        return {}


def reencode_video(input_path: str, output_path: str, engine: str = 'ffmpeg',
                   preset: str = 'h264_fast',
                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function for video reencoding"""
    pipeline = ReencodingPipeline(engine=engine, preset=preset)
    pipeline.load_model()
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)


def list_encoding_presets() -> Dict[str, Dict[str, Any]]:
    """List all available encoding presets"""
    return {
        'ffmpeg': REENCODING_PRESETS,
        'handbrake': HANDBRAKE_PRESETS
    }


def get_video_metadata(input_path: str) -> Dict[str, Any]:
    """Get video metadata using ffprobe"""
    pipeline = ReencodingPipeline()
    pipeline.load_model()
    return pipeline.get_video_info(input_path)