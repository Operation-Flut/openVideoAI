"""
FastAPI backend for openVideoAI
Serves the UI and provides REST API for model management
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes import models, processing, settings, health
from src.core.paths import ui_dir, project_root

# Initialize FastAPI app
app = FastAPI(title="openVideoAI", version=__version__)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules (router prefixes are relative, so no leading "/api")
app.include_router(models.router, prefix="/api")
app.include_router(processing.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(health.router, prefix="/api")

# Serve static UI files
_ui_path = ui_dir()
if _ui_path.exists():
    from fastapi.staticfiles import StaticFiles
    # Mount UI folder for direct access to CSS, JS, etc. at /ui/
    app.mount("/ui", StaticFiles(directory=str(_ui_path)), name="ui")

# Serve assets (images, icons, etc.)
_assets_path = project_root() / "assets"
if _assets_path.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=str(_assets_path)), name="assets")
