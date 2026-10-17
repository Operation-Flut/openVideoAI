# openVideoAI

<p align="center">
  <img src="assets/icon/black_glass.png" width="128"/>
</p>

Local-first toolkit for managing open-source image and video enhancement models.

---

> **Status:** Active development. Model discovery, downloads, UI configuration,
> multi-feature enhancement pipelines, and full export with codec/quality/audio/LUT
> control are all working.

## Overview

**openVideoAI** provides model discovery and downloads, a FastAPI backend, and a
web based UI for configuring image and video enhancement workflows. Multi-feature
pipelines (upscale, face recovery, denoise, sharpen, color, interpolation, …) run
end-to-end and write video with the codec, profile, bitrate, audio mode, container
and LUT you choose in the Export dialog.

## Screenshots
| Enhancment Page | Models Manager | Export Dialog |
| --- | --- | --- |
| ![Screenshot1](assets/Screenshot1.png) | ![Screenshot2](assets/Screenshot2.png) | *(screenshot pending)* |


## Features

- Browse enhancement models and manage local downloads.
- Build multi-step enhancement pipelines with per-step parameters.
- Export dialog: codec (H.264/H.265/VP9/AV1-NVENC/HEVC-NVENC), profile,
  Dynamic/Constant bitrate, quality level, container (MP4/MKV/MOV/WebM),
  audio mode (Copy/AAC/Opus/MP3/None), Apply LUT (.cube), image sequence output.
- Smart Select: auto-pick the best model for the loaded file (media, memory, install state).
- Settings persisted to disk (system resources, GPU backend, export defaults, appearance).
- Native desktop window via pywebview; also runs in a browser tab.

## Progress

| Area | Status |
|------|--------|
| Model catalog and model management | ✅ Available |
| Web UI and API | ✅ Available |
| Enhancement settings | ✅ Persisted (system, GPU, export, appearance, models, advanced) |
| Media inference and export | ✅ Working (all pipelines, real ffmpeg encoder, audio muxing, LUT) |
| CLI, batch processing, and presets | 🔄 Planned |

For details, see [Project status](docs/PROJECT_STATUS.md).

## GPU acceleration

- **NVIDIA CUDA** — detected automatically; use the *CUDA (NVIDIA)* backend.
- **AMD (Windows)** — install `torch-directml` to enable DirectML
  acceleration on AMD/Intel cards:

  ```bash
  pip install -r requirements-amd.txt
  ```

  Restart the app afterwards. The Settings page will enable the
  *AMD GPU (DirectML)* backend and list your adapter. Invalid/legacy device
  settings never reach PyTorch — an unusable device string falls back to the
  best available accelerator automatically.
- **AMD (Linux)** — a ROCm build of PyTorch is detected automatically
  (`hip` backend).
- **Apple Silicon** — Metal/MPS acceleration is enabled on Apple Silicon.

### Performance

- **Auto-tuned tiling.** Real-ESRGAN processes large frames in tiles; each
  tile is a separate GPU dispatch with fixed overhead (especially on
  DirectML). When the tile size is left at the shipped default (400 px) the
  app auto-upgrades it based on your GPU's VRAM (512 / 768 / 1024 px on
  4 / 8 / 12+ GB cards), which can cut tiling overhead several-fold. Pick an
  explicit tile size in Settings to override it.
- **Model choice matters.** For video, the light 6-block
  `RealESRGAN_x4plus_anime_6B` model runs roughly 3–4× faster than the full
  23-block `RealESRGAN` with a modest quality trade-off, and `RealESRGAN_x2plus`
  halves the output size when 2× is enough.
- **Half precision** (Settings → Enable FP16) roughly doubles speed on
  capable GPUs when the backend supports it.

## Model ecosystem

The catalog includes image and video restoration models for upscaling, face
recovery, denoising, sharpening, and frame interpolation, alongside color
transfer models such as NLUT, Deep Color Transfer, and Reinhard Transfer. It
also includes pure-math (non-AI) optical-flow algorithms for frame
interpolation (Farnebäck, DIS, Dual TV-L1, Lucas-Kanade) that run directly on
CPU with no model download. Farnebäck, DIS and Lucas-Kanade work out of the
box; Dual TV-L1 additionally needs `opencv-contrib-python`.

Each entry declares its compute requirements (`isAi`, `requiresCuda`,
`requiresMetal`, `canRunOnCpu`) in `models.json`. CUDA-required models can only
be downloaded on machines with a CUDA-capable GPU, and Metal-required models
only on Apple Silicon devices. Entries may also carry a `media` field
(`image`, `video` or `both`, the default) marking models that apply to only one
kind of media; the frame-interpolation and video-encoder entries are video-only.
The **Smart Select** switch in *Settings → Models* uses those fields, the detected
hardware and the memory budget in *Settings → System* to choose the best model for
the file being processed instead of the catalog default — see
[models.md](models.md#smart-select). Each entry also carries a `bestFor` note on
when to reach for it and which sibling to prefer instead; the model details popup
shows it, together with the media label, under *Best used for*.

See [models.md](models.md) for the complete catalog, comparisons, and usage
notes. Catalog entries describe intended integrations; they do not imply that
inference is implemented.
