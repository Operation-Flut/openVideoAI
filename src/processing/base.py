"""
Base processing utilities and error handling
"""
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for processing errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ModelLoadError(ProcessingError):
    """Failed to load model"""
    pass


class InferenceError(ProcessingError):
    """Inference failed"""
    pass


class UnsupportedFormatError(ProcessingError):
    """Unsupported file format"""
    pass


class ProcessingAborted(Exception):
    """Control-flow signal raised inside a running processing task when the
    application is shutting down (e.g. the desktop window was closed).

    It must propagate untouched through the processing pipeline so the worker
    can clean up partial output files and mark the task as aborted.
    """
    pass


class EnhancementPipeline:
    """Base class for enhancement pipelines"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._resolve_device(device)
        self._model = None
        self._model_name = None
        
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to a torch device type guaranteed to exist"""
        try:
            from ..utils.system_resources import resolve_default_device
            return resolve_default_device(device or 'auto')
        except Exception:
            # Conservative fallback if the system-resource helper is unavailable
            import torch
            if device == 'auto':
                try:
                    if torch.cuda.is_available():
                        return 'cuda'
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        return 'mps'
                except Exception:
                    pass
            return 'cpu'
    
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load model from path - override in subclass"""
        raise NotImplementedError
    
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process file - override in subclass"""
        raise NotImplementedError
    
    def _get_output_path(self, input_path: str, suffix: str, output_dir: Optional[str] = None) -> str:
        """Generate output path"""
        path = Path(input_path)
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"{path.stem}{suffix}{path.suffix}")
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def model_name(self) -> Optional[str]:
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value