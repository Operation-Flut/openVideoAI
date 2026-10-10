"""
Processing schemas for openVideoAI
"""
from typing import Optional, List
from pydantic import BaseModel


class UpscaleRequest(BaseModel):
    model_name: str = "Real-ESRGAN"
    scale: int = 4
    device: str = "auto"
    output_path: Optional[str] = None


class ProcessResponse(BaseModel):
    success: bool
    input_path: str
    output_path: str
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    scale_factor: Optional[int] = None
    model: Optional[str] = None
    frames_processed: Optional[int] = None
    frames_failed: Optional[int] = None
    total_frames: Optional[int] = None
    input_resolution: Optional[str] = None
    output_resolution: Optional[str] = None
    fps: Optional[float] = None
    error: Optional[str] = None
