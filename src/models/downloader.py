"""
Model Downloader for openVideoAI
Handles downloading AI models from remote URLs with progress tracking and verification.
"""
import os
import json
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from urllib.parse import urlparse
import threading
import time


@dataclass
class ModelInfo:
    """Model metadata from models.json"""
    name: str
    description: str
    size: str
    weight: str
    url: str
    download_url: Optional[str]
    description_short: str
    features: List[Dict[str, str]]
    is_ai: bool = True
    requires_cuda: bool = False
    requires_metal: bool = False
    can_run_on_cpu: bool = True

    @property
    def is_builtin(self) -> bool:
        """Built-in mathematical models (non-AI with no weights to download)."""
        return not self.is_ai and not self.download_url

    @property
    def filename(self) -> str:
        """Extract filename from download URL"""
        parsed = urlparse(self.download_url or "")
        return os.path.basename(parsed.path) or f"{self.name.lower().replace(' ', '_')}.pth"
    
    @property
    def local_path(self) -> Path:
        """Local path where model should be stored"""
        return Path("models") / self.filename


@dataclass
class DownloadProgress:
    """Progress information for a download"""
    model_name: str
    total_bytes: int
    downloaded_bytes: int
    speed_bps: float
    eta_seconds: float
    status: str  # 'downloading', 'completed', 'failed', 'verifying'
    error: Optional[str] = None


class ModelDownloader:
    """Handles downloading and managing AI models"""
    
    def __init__(self, models_dir: str = None, models_json: str = None):
        from src.core.paths import models_dir as _models_dir, models_json as _models_json
        self.models_dir = Path(models_dir) if models_dir is not None else _models_dir()
        self.models_json = Path(models_json) if models_json is not None else _models_json()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callbacks: List[Callable[[DownloadProgress], None]] = []
        self._cancel_flags: Dict[str, bool] = {}
        
    def add_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Register a callback for download progress updates"""
        self._progress_callbacks.append(callback)
        
    def _notify_progress(self, progress: DownloadProgress):
        """Notify all registered callbacks of progress update"""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception:
                pass  # Ignore callback errors
    
    def load_models_catalog(self) -> List[ModelInfo]:
        """Load model catalog from models.json"""
        with open(self.models_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        models = []
        seen_names = set()
        for model_data in data.get('models', []):
            # Skip duplicates (models.json has duplicates)
            if model_data['name'] in seen_names:
                continue
            seen_names.add(model_data['name'])
            
            models.append(ModelInfo(
                name=model_data['name'],
                description=model_data['description'],
                size=model_data['size'],
                weight=model_data['weight'],
                url=model_data['url'],
                download_url=model_data['downloadUrl'],
                description_short=model_data['descriptionShort'],
                features=model_data.get('features', []),
                is_ai=bool(model_data.get('isAi', True)),
                requires_cuda=bool(model_data.get('requiresCuda', False)),
                requires_metal=bool(model_data.get('requiresMetal', False)),
                can_run_on_cpu=bool(model_data.get('canRunOnCpu', True)),
            ))
        return models
    
    def get_downloaded_models(self) -> List[ModelInfo]:
        """Get list of models that are already downloaded"""
        catalog = self.load_models_catalog()
        downloaded = []
        for model in catalog:
            if model.is_builtin or model.local_path.exists():
                downloaded.append(model)
        return downloaded
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of models not yet downloaded"""
        catalog = self.load_models_catalog()
        available = []
        for model in catalog:
            if model.is_builtin:
                continue  # Built-in math models never need downloading
            if not model.local_path.exists():
                available.append(model)
        return available
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a specific model is downloaded"""
        catalog = self.load_models_catalog()
        for model in catalog:
            if model.name == model_name:
                return model.is_builtin or model.local_path.exists()
        return False
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name"""
        for model in self.load_models_catalog():
            if model.name == name:
                return model
        return None
    
    def download_model(self, model: ModelInfo, chunk_size: int = 8192) -> bool:
        """Download a single model with progress tracking"""
        # Built-in math/algorithms need no download
        if model.is_builtin:
            progress = DownloadProgress(
                model_name=model.name,
                total_bytes=0,
                downloaded_bytes=0,
                speed_bps=0,
                eta_seconds=0,
                status='completed',
                error=None,
            )
            self._notify_progress(progress)
            return True

        # Check if model has a valid download URL
        if not model.download_url:
            progress = DownloadProgress(
                model_name=model.name,
                total_bytes=0,
                downloaded_bytes=0,
                speed_bps=0,
                eta_seconds=0,
                status='failed',
                error=f"Model '{model.name}' does not have a direct download URL. Please visit {model.url} to download manually."
            )
            self._notify_progress(progress)
            return False
        
        self._cancel_flags[model.name] = False
        temp_path = self.models_dir / f".{model.filename}.tmp"
        response = None
        
        try:
            # Start download
            response = requests.get(model.download_url, stream=True, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').split(';', 1)[0].strip().lower()
            if content_type.startswith('text/') or content_type in {'application/json', 'application/xml'}:
                raise ValueError(
                    f"Download URL returned {content_type or 'text content'}, not a model file"
                )
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            start_time = time.time()
            last_update = start_time
            
            # Write to temporary file first
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if self._cancel_flags.get(model.name, False):
                        raise InterruptedError("Download cancelled")
                    
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Calculate speed and ETA
                        current_time = time.time()
                        elapsed = current_time - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        eta = (total_size - downloaded) / speed if speed > 0 else 0
                        
                        # Notify progress (throttle to ~10 updates/sec)
                        if current_time - last_update > 0.1 or downloaded == total_size:
                            progress = DownloadProgress(
                                model_name=model.name,
                                total_bytes=total_size,
                                downloaded_bytes=downloaded,
                                speed_bps=speed,
                                eta_seconds=eta,
                                status='downloading'
                            )
                            self._notify_progress(progress)
                            last_update = current_time
            
            # Verify download
            progress = DownloadProgress(
                model_name=model.name,
                total_bytes=total_size,
                downloaded_bytes=downloaded,
                speed_bps=0,
                eta_seconds=0,
                status='verifying'
            )
            self._notify_progress(progress)
            
            # Move temp file to final location
            final_path = self.models_dir / model.filename
            if final_path.exists():
                final_path.unlink()
            temp_path.rename(final_path)
            
            # Notify completion
            progress = DownloadProgress(
                model_name=model.name,
                total_bytes=total_size,
                downloaded_bytes=downloaded,
                speed_bps=0,
                eta_seconds=0,
                status='completed'
            )
            self._notify_progress(progress)
            
            return True
            
        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            
            progress = DownloadProgress(
                model_name=model.name,
                total_bytes=0,
                downloaded_bytes=0,
                speed_bps=0,
                eta_seconds=0,
                status='failed',
                error=str(e)
            )
            self._notify_progress(progress)
            return False
        finally:
            if response is not None:
                response.close()
    
    def cancel_download(self, model_name: str):
        """Cancel an ongoing download"""
        self._cancel_flags[model_name] = True
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model file"""
        model = self.get_model_by_name(model_name)
        if model:
            if model.is_builtin:
                return True  # Built-in math models cannot be deleted
            if model.local_path.exists():
                model.local_path.unlink()
                return True
        return False
    
    def verify_model(self, model: ModelInfo) -> bool:
        """Verify model file exists and has reasonable size"""
        if model.is_builtin:
            return True  # Built-in math models have nothing to verify

        path = model.local_path
        if not path.exists():
            return False
        
        # Check file size is reasonable (at least 1MB for these models)
        size = path.stat().st_size
        return size > 1_000_000


# ModelManager is defined in .manager and imported here for backward compatibility
from .manager import ModelManager


def get_model_files() -> List[str]:
    """Legacy function for backward compatibility"""
    models_dir = Path("models")
    if not models_dir.exists():
        return ["models"]
    
    files = ["models"]
    for filepath in models_dir.rglob("*"):
        if filepath.is_file():
            files.append(str(filepath))
    return files