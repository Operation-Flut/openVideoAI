"""
Model Downloader for openVideoAI
Handles downloading AI models from remote URLs with progress tracking and verification.
"""
import os
import json
import hashlib
import zipfile
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse
import threading
import time

logger = logging.getLogger(__name__)


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
    filename_src: Optional[str] = None
    # Extra weight files that belong to the same model (e.g. a neural model that
    # ships an encoder and a decoder as separate checkpoints). Each entry is a
    # dict with ``url`` and an optional ``filename`` (falls back to the URL
    # basename). Empty for single-file models.
    additional_downloads: List[Dict[str, str]] = field(default_factory=list)

    @property
    def is_builtin(self) -> bool:
        """Built-in mathematical models (non-AI with no weights to download)."""
        return not self.is_ai and not self.download_url

    @property
    def filename(self) -> str:
        """Filename used to store the downloaded payload locally.

        A catalog-provided ``filename`` wins (cloud hosts such as Google Drive
        and Dropbox rarely expose a meaningful name in the URL path). Otherwise
        fall back to the URL basename; drive-download URLs expose ``/download``
        as the path, so that is discarded too.
        """
        if self.filename_src:
            return self.filename_src
        parsed = urlparse(self.download_url or "")
        base = os.path.basename(parsed.path)
        if base in ("", "download", "."):
            # Keep a stable, meaningful local name for URLs without one.
            if "drive.google.com" in (parsed.netloc or ""):
                base = "google_drive_model.pth"
            else:
                base = f"{self.name.lower().replace(' ', '_')}.pth"
        return base

    @property
    def auxiliary_files(self) -> List[Dict[str, str]]:
        """Normalised list of extra ``{url, filename}`` weight files."""
        files: List[Dict[str, str]] = []
        for entry in self.additional_downloads:
            url = (entry or {}).get('url')
            if not url:
                continue
            filename = (entry.get('filename')
                        or os.path.basename(urlparse(url).path)
                        or f"{self.name.lower().replace(' ', '_')}_aux.pth")
            files.append({'url': url, 'filename': filename})
        return files

    @property
    def auxiliary_paths(self) -> List[Path]:
        """Local paths of the extra weight files."""
        return [Path("models") / f['filename'] for f in self.auxiliary_files]

    @property
    def all_paths(self) -> List[Path]:
        """Every local file that must exist for this model to be usable."""
        return [self.local_path, *self.auxiliary_paths]

    @property
    def all_filenames(self) -> List[str]:
        """Filenames of every local file that must exist for this model."""
        return [self.filename, *[f['filename'] for f in self.auxiliary_files]]

    @property
    def is_downloaded(self) -> bool:
        """True when the model is built-in or every weight file is present."""
        if self.is_builtin:
            return True
        return all(path.exists() for path in self.all_paths)

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
                filename_src=model_data.get('filename'),
                additional_downloads=list(model_data.get('additionalDownloads', []) or []),
            ))
        return models
    
    def get_downloaded_models(self) -> List[ModelInfo]:
        """Get list of models that are already downloaded"""
        catalog = self.load_models_catalog()
        downloaded = []
        for model in catalog:
            if model.is_downloaded:
                downloaded.append(model)
        return downloaded

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of models not yet downloaded"""
        catalog = self.load_models_catalog()
        available = []
        for model in catalog:
            if model.is_builtin:
                continue  # Built-in math models never need downloading
            if not model.is_downloaded:
                available.append(model)
        return available

    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a specific model is downloaded"""
        catalog = self.load_models_catalog()
        for model in catalog:
            if model.name == model_name:
                return model.is_downloaded
        return False
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name"""
        for model in self.load_models_catalog():
            if model.name == name:
                return model
        return None
    
    def download_model(self, model: ModelInfo, chunk_size: int = 8192) -> bool:
        """Download a model (all of its weight files) with progress tracking"""
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

        # A model may consist of several weight files (e.g. an encoder and a
        # decoder). Download every file and report a single cumulative progress
        # bar so multi-file models look like one download.
        targets = [{'url': model.download_url, 'filename': model.filename}]
        targets.extend(model.auxiliary_files)

        # Best-effort pre-flight so the progress bar knows the grand total.
        total_size = sum(self._probe_content_length(t['url']) for t in targets)

        downloaded_total = 0
        try:
            for target in targets:
                downloaded_total = self._download_single(
                    url=target['url'],
                    filename=target['filename'],
                    model_name=model.name,
                    downloaded_before=downloaded_total,
                    total_size=total_size,
                    chunk_size=chunk_size,
                )

            # Verify download
            progress = DownloadProgress(
                model_name=model.name,
                total_bytes=total_size or downloaded_total,
                downloaded_bytes=downloaded_total,
                speed_bps=0,
                eta_seconds=0,
                status='verifying'
            )
            self._notify_progress(progress)

            # Some downloadable models ship as archives (e.g. zip bundles of
            # several checkpoint files). Extract them into the models dir so
            # the weight files are directly usable, then drop the archive.
            self._extract_archive(self.models_dir / model.filename, model.name)

            # Notify completion
            progress = DownloadProgress(
                model_name=model.name,
                total_bytes=total_size or downloaded_total,
                downloaded_bytes=downloaded_total,
                speed_bps=0,
                eta_seconds=0,
                status='completed'
            )
            self._notify_progress(progress)

            return True

        except Exception as e:
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

    @staticmethod
    def _probe_content_length(url: str, timeout: int = 15) -> int:
        """Best-effort size lookup so multi-file downloads can show one bar."""
        try:
            response = requests.head(url, allow_redirects=True, timeout=timeout)
            if response.status_code == 200:
                return int(response.headers.get('content-length', 0) or 0)
        except Exception:
            pass
        return 0

    def _download_single(self, url: str, filename: str, model_name: str,
                         downloaded_before: int, total_size: int,
                         chunk_size: int = 8192) -> int:
        """Stream one file to disk, returning the cumulative byte count."""
        temp_path = self.models_dir / f".{filename}.tmp"
        final_path = self.models_dir / filename
        response = None
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').split(';', 1)[0].strip().lower()
            if content_type.startswith('text/') or content_type in {'application/json', 'application/xml'}:
                raise ValueError(
                    f"Download URL returned {content_type or 'text content'}, not a model file"
                )

            file_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            start_time = time.time()
            last_update = start_time

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if self._cancel_flags.get(model_name, False):
                        raise InterruptedError("Download cancelled")

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        current_time = time.time()
                        elapsed = current_time - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        grand_total = total_size or (downloaded_before + file_size)
                        remaining = max(grand_total - (downloaded_before + downloaded), 0)
                        eta = remaining / speed if speed > 0 else 0

                        # Notify progress (throttle to ~10 updates/sec)
                        if current_time - last_update > 0.1 or (file_size and downloaded == file_size):
                            progress = DownloadProgress(
                                model_name=model_name,
                                total_bytes=grand_total,
                                downloaded_bytes=downloaded_before + downloaded,
                                speed_bps=speed,
                                eta_seconds=eta,
                                status='downloading'
                            )
                            self._notify_progress(progress)
                            last_update = current_time

            if final_path.exists():
                final_path.unlink()
            temp_path.rename(final_path)
            return downloaded_before + downloaded

        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _is_torch_checkpoint(path: Path) -> bool:
        """torch.save() archives are zip files; detect their marker entry."""
        try:
            with zipfile.ZipFile(path) as zf:
                return any(name.startswith("archive/") for name in zf.namelist())
        except zipfile.BadZipFile:
            return False

    def _extract_archive(self, path: Path, model_name: str) -> bool:
        """Extract a downloaded zip payload into the models directory.

        Torch checkpoints are also zip containers (they hold an ``archive/``
        member), so those are left untouched. Any other zip is treated as a
        model bundle (e.g. Deep Color Transfer's ``checkpoints.zip``), extracted
        and deleted so only the real weight files remain.
        """
        if not zipfile.is_zipfile(path):
            return False
        if self._is_torch_checkpoint(path):
            return False
        try:
            models_root = self.models_dir.resolve()
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    target = (models_root / info.filename).resolve()
                    if not str(target).startswith(str(models_root) + os.sep) and target != models_root:
                        logger.warning(f"Skipping unsafe archive member: {info.filename}")
                        continue
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            path.unlink()
            logger.info(f"Extracted archive for '{model_name}' into {self.models_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to extract archive for '{model_name}': {e}")
            return False

    def cancel_download(self, model_name: str):
        """Cancel an ongoing download"""
        self._cancel_flags[model_name] = True
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model file (and all of its weight files)"""
        model = self.get_model_by_name(model_name)
        if model:
            if model.is_builtin:
                return True  # Built-in math models cannot be deleted
            # Legacy archive bundles that are not declared through
            # additionalDownloads still ship more than one weight file.
            companion_names = {
                "Deep Color Transfer": ("latest_net_G_A.pth", "latest_net_C_A.pth"),
            }.get(model.name, ())
            removed = False
            for filename in (*model.all_filenames, *companion_names):
                path = self.models_dir / filename
                if path.exists():
                    path.unlink()
                    removed = True
            return removed
        return False

    def verify_model(self, model: ModelInfo) -> bool:
        """Verify model files exist and have reasonable size"""
        if model.is_builtin:
            return True  # Built-in math models have nothing to verify

        primary = model.local_path
        if not primary.exists():
            return False
        # Multi-file models must have every weight file present too.
        if not all(path.exists() for path in model.auxiliary_paths):
            return False

        # Check file size is reasonable (at least 1MB for these models)
        size = primary.stat().st_size
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