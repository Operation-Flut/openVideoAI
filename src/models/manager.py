"""
Model Manager for openVideoAI
High-level management combining downloader, catalog loading, and initialization.
"""
from pathlib import Path
from typing import Dict, List, Optional, Callable
from .downloader import ModelDownloader, ModelInfo, DownloadProgress


class ModelManager:
    """High-level model management combining downloader and initialization"""
    
    def __init__(self, models_dir: str = "models", models_json: str = "models.json"):
        self.downloader = ModelDownloader(models_dir, models_json)
        self._initialization_complete = False
    
    def initialize(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict:
        """Initialize model directory and verify existing models"""
        if progress_callback:
            progress_callback("Creating models directory...", 0.1)
        
        self.downloader.models_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback("Loading model catalog...", 0.3)
        
        catalog = self.downloader.load_models_catalog()
        
        if progress_callback:
            progress_callback("Verifying downloaded models...", 0.5)
        
        downloaded = self.downloader.get_downloaded_models()
        available = self.downloader.get_available_models()
        
        # Verify each downloaded model
        verified = []
        for i, model in enumerate(downloaded):
            if progress_callback:
                progress = 0.5 + 0.4 * (i / len(downloaded)) if downloaded else 0.9
                progress_callback(f"Verifying {model.name}...", progress)
            
            if self.downloader.verify_model(model):
                verified.append(model)
        
        if progress_callback:
            progress_callback("Initialization complete", 1.0)
        
        self._initialization_complete = True
        
        return {
            'total_models': len(catalog),
            'downloaded': len(verified),
            'available': len(available),
            'verified_models': verified,
            'available_models': available
        }
    
    def get_model_status(self) -> Dict:
        """Get current model status"""
        catalog = self.downloader.load_models_catalog()
        downloaded = self.downloader.get_downloaded_models()
        available = self.downloader.get_available_models()

        return {
            'total': len(catalog),
            'downloaded': len(downloaded),
            'available': len(available),
            'models': [
                {
                    'name': m.name,
                    'description': m.description_short,
                    'size': m.size,
                    'weight': m.weight,
                    'downloaded': m in downloaded,
                    'features': [f['name'] for f in m.features],
                    'is_ai': m.is_ai,
                    'is_builtin': m.is_builtin,
                    'requires_cuda': m.requires_cuda,
                    'requires_metal': m.requires_metal,
                    'can_run_on_cpu': m.can_run_on_cpu,
                    'has_download_url': bool(m.download_url),
                }
                for m in catalog
            ]
        }
    
    def download_model(self, model_name: str, progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> bool:
        """Download a specific model"""
        model = self.downloader.get_model_by_name(model_name)
        if not model:
            return False
        
        if progress_callback:
            self.downloader.add_progress_callback(progress_callback)
        
        return self.downloader.download_model(model)
    
    def ensure_model(self, model_name: str, progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> bool:
        """Ensure a model is available locally, downloading if necessary"""
        if self.downloader.is_model_downloaded(model_name):
            return True
        return self.download_model(model_name, progress_callback)
