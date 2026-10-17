"""
FastAPI application factory.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .routes import models, processing, settings, health, update
from ..core.config import __version__
from ..core.paths import project_root, exports_dir, temp_processing_dir
from ..events.bus import get_event_bus
from ..services.model_service import get_model_service
from ..services.processing_service import get_processing_service
from ..services.settings_service import get_settings_service

# Import startup progress tracking from dedicated module
from .startup import get_startup_progress, set_startup_progress, mark_startup_complete


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print(f"Starting openVideoAI v{__version__}")
    set_startup_progress("initializing", 5, "Initializing core services...")

    # Initialize services
    get_model_service()
    set_startup_progress("initializing", 15, "Initializing model service...")
    get_model_service()
    
    set_startup_progress("initializing", 25, "Initializing processing service...")
    get_processing_service()
    
    set_startup_progress("initializing", 35, "Initializing settings service...")
    get_settings_service()
    
    set_startup_progress("initializing", 45, "Initializing event bus...")
    get_event_bus()

    # Ensure directories exist
    set_startup_progress("initializing", 55, "Setting up directories...")
    exports_dir().mkdir(parents=True, exist_ok=True)
    temp_processing_dir().mkdir(parents=True, exist_ok=True)

    # Refresh model registry
    set_startup_progress("initializing", 70, "Loading model registry...")
    model_service = get_model_service()
    model_service.get_all_models_with_status()

    set_startup_progress("initializing", 90, "Finalizing startup...")
    mark_startup_complete()

    yield

    # Shutdown
    print("Shutting down openVideoAI")
    processing_service = get_processing_service()
    processing_service.request_shutdown()

    model_service = get_model_service()
    model_service.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="openVideoAI",
        version=__version__,
        description="Local AI Video Enhancement Toolkit API",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers under /api so they line up with the UI's API client
    # (axios baseURL = "/api") and so client-side routes like /models and
    # /settings can be served by the SPA without colliding with the API.
    app.include_router(models.router, prefix="/api")
    app.include_router(processing.router, prefix="/api")
    app.include_router(settings.router, prefix="/api")
    app.include_router(health.router, prefix="/api")
    app.include_router(update.router, prefix="/api")

    # Static files for UI (if built)
    ui_dist = project_root() / "UI" / "dist"
    if ui_dist.exists():
        # Serve built assets and icons from their absolute (Vite default) paths.
        assets_dir = ui_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        icon_dir = ui_dist / "icon"
        if icon_dir.exists():
            app.mount("/icon", StaticFiles(directory=icon_dir), name="icon")

        # Keep the /ui mount for backward compatibility.
        app.mount("/ui", StaticFiles(directory=ui_dist, html=True), name="ui")

        # SPA entry point at the root so desktop/browser modes load the app.
        # Registered after the API routers, so /api/*, /docs, /openapi.json,
        # /favicon.ico, ... still resolve to their handlers first. Anything
        # unmatched falls back to index.html so client-side routes
        # (e.g. /models, /settings) work on a hard refresh.
        _index_file = ui_dist / "index.html"

        @app.get("/", include_in_schema=False)
        async def spa_root():
            return FileResponse(_index_file)

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            # Unknown API paths should 404 rather than silently returning HTML.
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            candidate = ui_dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(_index_file)
    else:
        # Root endpoint (API-only) when the UI has not been built yet.
        @app.get("/")
        async def root():
            return {
                "name": "openVideoAI",
                "version": __version__,
                "description": "Local AI Video Enhancement Toolkit",
                "docs": "/docs",
                "health": "/health",
            }

        @app.get("/api/startup-progress", include_in_schema=False)
        async def startup_progress():
            """Get startup progress for loading screen."""
            return get_startup_progress()

    # Favicon
    @app.get("/favicon.ico")
    async def favicon():
        icon_path = project_root() / "assets" / "icon" / "black_glass.ico"
        if icon_path.exists():
            return FileResponse(icon_path)
        return {"message": "No favicon"}

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)