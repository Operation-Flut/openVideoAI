# Project status

openVideoAI is an early-stage foundation for local image and video enhancement.
The application currently provides model discovery and local download
management, a FastAPI backend, and browser interfaces for managing models and
configuring enhancement options. The model catalog is documented in
[models.md](../models.md).

## Progress

| Area | Status | Current state |
|------|--------|---------------|
| Model catalog | Complete | Catalog metadata covers 11 unique models and enhancement features. |
| Model management | Complete | The UI and API support browsing models and managing local downloads. |
| Web interface and API | In progress | FastAPI routes and browser interfaces are available for model management and enhancement settings. |
| Enhancement configuration | In progress | The UI collects options for feature areas, including color matching and LOG restoration. |
| Media inference | Complete | End-to-end image/video processing works (Reinhard + WCT2 neural; NLUT/DCT broken-checkpoint fail-fast). Output files produced. |
| Processed media export | Complete | Output files written to output directory; endpoint returns result metadata. |
| Color Matching (Match Color) | Complete | Reinhard built-in + WCT2 neural implemented; model routing, strength/preserve_luminance, reference required. NLUT/DCT registered but non-functional (checkpoint issues). |
| Command-line interface | Planned | A CLI for running enhancement workflows from a terminal is not available yet. |
| Batch processing | Planned | Processing multiple files in a single workflow is not available yet. |
| Presets | Planned | Saveable and reusable enhancement presets are not available yet. |
| MCP server | Planned | An MCP server for controlling the application with AI tools is not implemented; the current server module is a placeholder. |
| Reencoding through FFMPEG and Handbgrakes engines | planned | not yet started with |
| Overexposure & Contrast Fix | planned | not yet implemented, usable Models exist in catalog |
| Project File Saving (.avp (AI Video Project) File) | planned | not yest started |
| Dedicated Motion deblur | not implemented | Necessary Models are added, but no dedicated Function for this application implemented |
| Fix compression | planned | not yet started |
| FCK AI Mode (when enabeled it disabeles any AI Model and its features so the app purely relies on NON-AI, math, algorithem, etc models)| planned | Necessary Models are implemented |
| JXL (JPEG XL), avif Image Format support | planned | not started |


## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List models with download status |
| GET | `/api/models/{name}` | Get model details |
| POST | `/api/models/download` | Start a model download |
| GET | `/api/models/{name}/progress` | Get download progress |
| POST | `/api/models/{name}/cancel` | Cancel an active download |
| GET | `/api/status` | Get system status |
| GET | `/api/settings` | Get persisted application settings |
| POST | `/api/settings` | Update system resources, GPU modes, export, and appearance |
| POST | `/api/settings/model-dir` | Update model storage directory |
| POST | `/api/settings/export-dir` | Update standard export output directory |
| GET | `/api/system/info` | Get hardware capabilities and runtime environment |
| GET | `/api/system/gpus` | List detected GPU accelerators and compute devices |
| DELETE | `/api/models/cache` | Clear cache and temporary processing files |
| POST | `/api/models/verify` | Verify downloaded model integrity |

Interactive API documentation is available at `/docs` while the app is running.


## Implementation notes

- Model and feature catalog entries describe intended integrations; they do
  not guarantee that a model can currently run inside the application.
- Enhancement controls execute inference (Reinhard + WCT2 neural; NLUT/DCT fail-fast due to broken checkpoints). Color Matching requires reference image; missing ref raises clear error.
- Model sizes are estimates, not verified download sizes or performance
  benchmarks. Upstream projects and checkpoints may have separate licenses.
- Performance will depend on the eventual inference implementation, model
  checkpoint, execution backend, input media, and available hardware.


## Default Presets
[ ] 60fps
[ ] Super SloMo
[ ] Frame Restoration
[ ] Restoration
[ ] 2k
[ ] 4k
[ ] Color Normalization
[ ] Color Restoration
[ ] All good All Fine
[ ] Deblur
[ ] "Fix da Focus"
[ ] Motion-de-blur
[ ] "Steal da Grade"