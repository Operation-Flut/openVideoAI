# Model catalog

openVideoAI keeps metadata for open-source image and video restoration models in
[models.json](models.json). The catalog describes potential model integrations;
it is not a claim that every model is already runnable in the application.

It currently lists **52 unique models**: **40** have a catalog download URL and
**12** do not. Nine of the twelve without a URL are non-AI "built-in"
algorithms (FFmpeg, the classical optical-flow fallbacks, and the classical
colour/light operators) that run without any weights; the other three are
metadata-only. The model manager skips repeated names in the catalog.

> **Current status:** **22** of the 52 entries have a runtime implementation in
> this build; the other **30** are catalog metadata, downloadable assets, or
> models whose code/runtime is not integrated. Those 30 are reported as
> *Implemented: No* in the model details popup and are never offered as pipeline
> steps. Runnable entry points are Real-ESRGAN, Real-CUGAN,
> RealESRGAN_x4plus_anime_6B and RealESRGAN_x2plus (upscaling); CodeFormer,
> GFPGAN and RestoreFormer (faces); SCUNet, NAFNet-SIDD, NAFNet-REDS and
> NAFNet-GoPro (denoise); NAFNet-GoPro, NAFNet-REDS and Restormer (sharpen /
> deblur and focus); NLUT, WCT2 and Reinhard Transfer (colour match); Zero-DCE
> and the classical low-light operator (low light); the classical Reinhard tone
> mapper (SDR to HDR); the classical optical-flow fallbacks (frame
> interpolation); and FFmpeg (re-encoding). Treat listed sizes as estimates, not
> verified benchmarks.

## Catalog models

| Model | Use | Feature(s) | Approx. catalog size | Type | Catalog hardware | Runs in this build |
| --- | --- | --- | --- | --- | --- | --- |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Fast, general-purpose image upscaling and texture enhancement. | Upscale | ~65MB | AI | CUDA | Yes |
| [CodeFormer](https://github.com/sczhou/CodeFormer) | Advanced face restoration for restoring heavily degraded or blurred facial details. | Face Recovery | ~360MB | AI | CPU-capable | Yes |
| [NAFNet-SIDD](https://github.com/megvii-research/NAFNet) | Denoises images while preserving fine details. | Denoise | ~464MB | AI | CPU-capable | Yes |
| [NAFNet-REDS](https://github.com/megvii-research/NAFNet) | General video restoration for blur and complex artifacts. | Sharpen / Deblur, Denoise | ~260MB | AI | CPU-capable | Yes |
| [NAFNet-GoPro](https://github.com/megvii-research/NAFNet) | Deblurs images affected by camera shake and fast motion. | Sharpen / Deblur, Denoise | ~272MB | AI | CPU-capable | Yes |
| [SCUNet](https://github.com/cszn/SCUNet) | Removes camera grain and JPEG artifacts while preserving edges. | Denoise | ~45MB | AI | CPU-capable | Yes |
| [RIFE](https://github.com/vladmandic/rife) | Fast video frame interpolation and smooth slow-motion generation. | Frame Interpolation, SloMo | ~20MB | AI | CPU-capable | No (needs `rife_arch`) |
| [RIFE v4.26 (PyTorch)](https://huggingface.co/thornmaze/RIFE) | RIFE v4.26 PyTorch model archive for video frame interpolation. | Frame Interpolation, SloMo | ~23MB | AI | CPU-capable | No (needs `rife_arch`) |
| [RIFE 4.25 (MLX)](https://huggingface.co/mlx-community/RIFE-4.25) | Apple Silicon MLX weights for RIFE 4.25 frame interpolation. | Frame Interpolation, SloMo | ~23MB | AI | Apple Silicon / Metal | No (metadata only) |
| [WMS-RIFE v3 (PyTorch)](https://huggingface.co/Anson-Saju-George/wms-rifev3) | Domain-specific RIFE v3 checkpoint for weather and satellite video. | Frame Interpolation, SloMo | ~12MB | AI | CPU-capable | No (metadata only) |
| [RIFE (PyTorch Safetensors)](https://huggingface.co/TensorForger/RIFE-safetensors) | RIFE v4 weights in PyTorch-compatible safetensors format. | Frame Interpolation, SloMo | ~12MB | AI | CUDA | No (metadata only) |
| [RIFE v4.9 (ONNX)](https://huggingface.co/edgetools/rife) | RIFE v4.9 ONNX model for video frame interpolation. | Frame Interpolation, SloMo | ~21MB | AI | CPU-capable | No (metadata only) |
| [FILM](https://github.com/google-research/frame-interpolation) | High-quality frame interpolation optimized for handling large motion and gaps. | Frame Interpolation, Frame Recovery | ~130MB | AI | CPU-capable | No (needs `film_arch`) |
| [Farnebäck Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) | Dense polynomial optical flow for mathematical frame interpolation. | Frame Interpolation, SloMo | Built-in (&lt; 1MB) | Non-AI (math) | CPU | Yes |
| [DIS Optical Flow](https://docs.opencv.org/4.x/da/d06/classcv_1_1DISOpticalFlow.html) | Ultra-fast variational mathematical optical flow for real-time motion interpolation. | Frame Interpolation, SloMo | Built-in (&lt; 1MB) | Non-AI (math) | CPU | Yes |
| [Dual TV-L1 Optical Flow](https://docs.opencv.org/4.x/dc/d47/classcv_1_1DualTVL1OpticalFlow.html) | High-precision total-variation mathematical optical flow preserving sharp motion boundaries. | Frame Interpolation, SloMo | Built-in (&lt; 1MB) | Non-AI (math) | CPU | No (needs `optflow`) |
| [Lucas-Kanade (Pyramidal LK)](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) | Classical pyramidal differential optical flow for motion estimation and interpolation. | Frame Interpolation, SloMo | Built-in (&lt; 1MB) | Non-AI (math) | CPU | Yes |
| [RAFT Optical Flow](https://github.com/princeton-vl/RAFT) | Deep learning recurrent optical flow for high-accuracy motion estimation. | Frame Interpolation, SloMo | ~20MB | AI | CUDA | No (needs `raft_arch`) |
| [FlowFormer Optical Flow](https://github.com/drinkingcoder/FlowFormer) | Transformer-based optical flow with global attention for complex motion. | Frame Interpolation, SloMo | ~40MB | AI | CUDA | No (needs `flowformer_arch`) |
| [GMA Optical Flow](https://github.com/zacjiang/GMA) | Global motion aggregation for robust optical flow in complex scenes. | Frame Interpolation, SloMo | ~25MB | AI | CUDA | No (needs `gma_arch`) |
| [SPyNet Optical Flow](https://github.com/princeton-vl/RAFT) | Lightweight spatial pyramid network for efficient optical flow. | Frame Interpolation, SloMo | ~5MB | AI | CPU-capable | No (needs `spynet_arch`) |
| [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) | Multi-frame video restoration for cleaning noisy, low-light, or degraded LOG footage. | Multipurpose Restoration, Upscale, Log Restoration | ~200MB | AI | CPU-capable | No (needs `realbasicvsr_arch`) |
| [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | Multipurpose 2D/anime image upscaling, denoising, and artifact removal. | Multipurpose Pipeline, Upscale | ~40MB | AI | CPU-capable | Yes |
| [NLUT](https://github.com/semchan/NLUT) | Extracts colors from a reference image and creates a fast 3D LUT. | Match Color | ~225MB | AI | CPU-capable | Yes |
| [Deep Color Transfer](https://github.com/codeslake/Color_Transfer_Histogram_Analogy) | Advanced, semantically-aware color transfer for complex lighting. | Match Color | ~153MB | AI | CPU-capable | No (needs `deepcolor_arch`) |
| [WCT2](https://github.com/clovaai/WCT2) | Neural color matching that reproduces the reference's look. | Match Color | ~40MB | AI | CPU-capable | Yes |
| [Reinhard Transfer](https://github.com/jrosebr1/color_transfer) | Instant, zero-VRAM mathematical color matching. | Match Color | &lt; 1MB | Non-AI (math) | CPU | Yes |
| [Sapsan-VFI](https://huggingface.co/SnJake/Sapsan-VFI) | Deep learning model for up to 2x video frame interpolation and slow-motion generation. | Frame Interpolation, SloMo | ~270MB | AI | CPU-capable | No (metadata only) |
| [BSRGAN](https://github.com/cszn/BSRGAN) | Blind restoration + 4x upscaling that handles blur, noise, and JPEG artifacts. | Upscale, Multipurpose Restoration | ~67MB | AI | CPU-capable | No (metadata only) |
| [GFPGAN](https://github.com/TencentARC/GFPGAN) | Face-focused restoration for blurry, noisy, or low-resolution faces. | Face Recovery | ~333MB | AI | CPU-capable | Yes |
| [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) | Face restoration with a transformer decoder for fine detail preservation. | Face Recovery | ~443MB | AI | CPU-capable | Yes |
| [RestoreFormer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus) | VQGAN + full-attention face restoration with finer skin texture. | Face Recovery | ~458MB | AI | CPU-capable | No (needs `restoreformer_plusplus_arch`) |
| [DifFace](https://github.com/czczup/DifFace) | Diffusion face restoration focused on identity preservation. | Face Recovery | ~639MB | AI | CPU-capable | No (needs `difface_arch`) |
| [Restormer](https://github.com/swz30/Restormer) | Transformer defocus-deblurring for out-of-focus images. | Sharpen / Deblur, Focus Fix | ~100MB | AI | CPU-capable | Yes |
| [DeblurGAN-v2](https://github.com/VITA-Group/DeblurGANv2) | GAN motion deblurring (FPN backbone); weights require manual download. | Sharpen / Deblur | ~157MB | AI | CPU-capable | No (needs `tensorflow`) |
| [MPRNet](https://github.com/swz30/MPRNet) | Multi-stage CNN deblurring (GoPro); weights require manual download. | Sharpen / Deblur, Denoise | ~57MB | AI | CPU-capable | No (needs `mprnet_arch`) |
| [FFTformer](https://github.com/kkkls/fftformer) | Frequency-domain transformer for motion deblurring. | Sharpen / Deblur | ~66MB | AI | CPU-capable | No (needs `fftformer_arch`) |
| [DiffBIR](https://github.com/XPixelGroup/DiffBIR) | Diffusion prior for high-quality blind restoration (GPU with 12GB+ VRAM). | Multipurpose Restoration, Upscale | ~6.4GB | AI | CUDA | No (metadata only) |
| [SUPIR](https://github.com/Fanghua-Yu/SUPIR) | SDXL-based generative restoration for photo-realistic results (needs ~24GB VRAM). | Multipurpose Restoration, Upscale | ~5.3GB | AI | CUDA | No (metadata only) |
| [SeeSR](https://github.com/cswry/SeeSR) | Semantics-aware diffusion super-resolution with caption guidance. | Upscale | ~4.5GB | AI | CUDA | No (metadata only) |
| [StableSR](https://github.com/IceClear/StableSR) | Diffusion super-resolution with time-aware encoding for realistic detail. | Upscale | ~6.5GB | AI | CUDA | No (metadata only) |
| [CCSR](https://github.com/csslc/CCSR) | Controllable diffusion super-resolution with structure/texture balance. | Upscale | ~7.3GB | AI | CUDA | No (metadata only) |
| [OSEDiff](https://github.com/cswry/OSEDiff) | One-step diffusion super-resolution (fast, NeurIPS 2024); weights via manual download. | Upscale | ~1.3GB | AI | CUDA/ROCm/DirectML | No (metadata only) |
| [TSRN-RTVD](https://github.com/msu-video-group/tsrn-rtvd) | Real-time video deblurring (RTVD); streaming API via the tsrn-rtvd package. | Sharpen / Deblur | ~57MB | AI | CPU-capable | No (needs `tsrn_arch`) |
| [OpenCV NAFNet (ONNX)](https://github.com/opencv/opencv_zoo) | NAFNet GoPro deblurring as ONNX for OpenCV / cv2.dnn. | Sharpen / Deblur | ~92MB | AI | CPU-capable | No (metadata only) |
| [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) | Tiny AI low-light enhancer that lifts shadows while keeping colors natural. | Low-Light Enhancement | ~3MB | AI | CPU-capable | Yes |
| Auto Low-Light (Classical) | Instant classical low-light boost with no download. | Low-Light Enhancement | &lt; 1MB | Non-AI (math) | CPU | Yes |
| HDR (Reinhard Tone Mapping) | Instant SDR to HDR conversion with Reinhard tone mapping. | SDR to HDR | &lt; 1MB | Non-AI (math) | CPU | Yes |
| [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN) | Anime/line-art upscaling with a lighter 6-block network. | Upscale, Multipurpose Pipeline | ~18MB | AI | CUDA | Yes |
| [RealESRGAN_x2plus](https://github.com/xinntao/Real-ESRGAN) | 2x upscaling for photos and video. | Upscale, Video Super Resolution | ~64MB | AI | CUDA | Yes |
| [ffmpeg](https://ffmpeg.org/) | FFmpeg-based re-encoding (H.264/H.265/AV1). | Reencoding | Built-in (&lt; 1MB) | Non-AI (math) | CPU | Yes |
| [handbrake](https://handbrake.fr/) | HandBrake CLI re-encoding (no runtime in this build). | Reencoding | Built-in (&lt; 1MB) | Non-AI (math) | CPU | No (metadata only) |


The hardware column reflects the catalog's declared requirements and CPU
fallback, not a guarantee that the app currently implements that runtime or
can run every listed model on that device. See
[Downloads and model files](#downloads-and-model-files) before downloading.

The catalog also defines **21** features. Thirteen have a runnable model,
three (**Frame Recovery**, **Multipurpose Restoration**, **Log Restoration**)
only have metadata-only entries, and five (**Correct Distortion**,
**Colorization**, **Object Removal**, **Overexposure & Contrast Fix**, **Flat
Color Profile**) are declared with no model entry at all. These labels describe
intended use and catalog/UI availability; they do not imply that the
corresponding inference implementations are complete. A feature is only offered
on the Enhance page when it has at least one runnable, available model.

## Choosing a model

- For general image upscaling, start with Real-ESRGAN; Real-CUGAN and the
  smaller Real-ESRGAN variants are alternatives once downloaded.
- For faces, consider CodeFormer, GFPGAN, or RestoreFormer (all three run in
  this build).
- For sensor noise, consider SCUNet or NAFNet-SIDD; for motion blur, consider
  Restormer or NAFNet-GoPro. All four run in this build. The NAFNet entries load
  their width-64 checkpoints with the block layout declared in the catalog.
- For anime or line art, consider Real-CUGAN.
- For generating intermediate video frames, consider RIFE or FILM. No AI
  interpolation model is integrated in this build, so the frame-interpolation
  pipeline currently offers only the classical optical-flow fallbacks below.
- The catalog includes the non-AI optical-flow entries Farnebäck, DIS, Dual
  TV-L1, and Lucas-Kanade (Pyramidal LK). They are marked as built-in,
  CPU-capable algorithms and do not need downloaded model weights. Farnebäck,
  DIS and Lucas-Kanade are wired into the frame-interpolation pipeline; Dual
  TV-L1 additionally requires `opencv-contrib-python` (the `cv2.optflow`
  module) and is only offered when that is installed.
- The catalog also lists the deep-learning optical-flow entries RAFT,
  FlowFormer, GMA, and SPyNet. Their hardware requirements are metadata; the
  entries do not currently provide app-level optical-flow inference.
- The catalog includes RIFE checkpoints in several formats and variants:
  Practical-RIFE v4.26 as a ZIP archive, RIFE 4.25 for Apple's MLX runtime, a
  weather/satellite-specialized PyTorch v3 checkpoint, PyTorch weights in
  safetensors format, and RIFE v4.9 in ONNX format. Each needs compatible model
  code and a matching runtime; downloading a checkpoint alone does not add
  interpolation to the app.
- For video restoration, consider RealBasicVSR; it is metadata-only in this
  build (its runtime is not bundled).
- For reference-based colour matching, the catalog lists NLUT, WCT2, Reinhard
  Transfer, and Deep Colour Transfer. NLUT, WCT2 and Reinhard Transfer all run
  in this build; Deep Colour Transfer is metadata-only.

## LOG footage module

The catalog contains a **LOG Footage Enhancer & Neural Color Grade** module
configuration for footage such as S-Log, C-Log, V-Log, LogC, and D-Log. Its
proposed stages are temporal denoising, exposure and contrast recovery, and
neural tone mapping/colour restoration.

This is configuration-only and currently inert: the catalog's `modules` block is
not exposed through the API and no part of the UI reads it, so there is no LOG
control panel and no Rec.709/HDR export. The **Log Restoration** feature appears
in the catalog with a single metadata-only model. Treat all of this as a design
sketch, not an implemented colour-management pipeline.

Results, processing time, memory use, and hardware support depend on the model
checkpoint, implementation, input media, and execution backend. Compare outputs
on representative files before choosing a production workflow.

## Downloads and model files

Each catalog entry contains an upstream project URL in
[models.json](models.json); `downloadUrl` is optional. For entries with a
download URL, it may point to a release or project page rather than directly
to a checkpoint. Entries without a download URL cannot be downloaded through
the model manager; visit the upstream project for availability and manual
download instructions. A model with no download URL is not necessarily
built-in: only non-AI entries without a download URL are treated as built-in.
Downloading a file alone does not install its runtime, dependencies, or
inference code.

### AI, non-AI, and hardware requirements

Each entry can declare the following fields in `models.json`:

- `isAi` — `true` for neural-network models, `false` for pure-math algorithms.
- `requiresCuda` — `true` when the catalog requires an NVIDIA CUDA-capable GPU.
- `requiresMetal` — `true` when the catalog requires Apple Silicon (Metal).
- `canRunOnCpu` — whether the model can fall back to CPU execution.
- `arch` — the runtime dispatch key. `null` marks a metadata-only entry: it is
  listed (and can be downloaded where a URL exists) but has no implementation in
  this build, so the API reports `implemented: false` and it is never offered as
  a pipeline step.
- `requiresModule` — an optional Python module the entry's runtime needs. The
  model is reported as not implemented when that module is missing from the
  current build.

The catalog is the single source of truth for these requirements:

- CUDA-required models are only downloadable (and visible as compatible) on
  machines with a detected CUDA-capable GPU.
- Metal-required models are only downloadable on Apple Silicon devices.
- Non-AI entries with no download URL are treated as **built-in**: they are
  reported as already available, need no download, and appear in the UI filter
  "Non-AI (Pure Math)". Built-ins run from code rather than from a checkpoint,
  so they report no `local_path`.

### Experimental features

A catalog entry — a feature in `features[]` or a model in `models[]` — can carry
`"experimental": true`. Experimental items are treated as not production-ready:
they are hidden from the Enhance page and skipped during processing unless
**Settings → Advanced → Experimental Features** is switched on. The small ⓘ
next to that toggle lists whatever is currently marked; the list is read from
this catalog, so it never drifts.

Currently marked as experimental (each is verified to fail or to fall short of its
promise in the current build):

| Feature | Why |
| --- | --- |
| Frame Interpolation | Every AI interpolation model (RIFE, FILM, RAFT, SPyNet, …) is unavailable in this build, so only the classical optical-flow fallbacks run. |
| SDR to HDR | No AI inverse-tone-mapping model is integrated. The classical entry is a real 10-bit conversion (PQ for HDR10/HDR10+, HLG) but there is no learned model behind it. |

Two features were previously marked and are now unmarked because the underlying
defects were fixed and re-verified end to end:

| Feature | What was broken |
| --- | --- |
| Match Color | The reference image never reached the pipeline (`MatchColorPipeline` did not accept a `parameters` dict, so `instantiate_pipeline()` dropped `reference_path`), a stale `ref_path` reference made every image run fail after writing its output, and the AI models could not load: the pipeline imported names (`NLUT`, `WCT2Encoder`) that the arch modules do not export, and the WCT2 branch re-implemented a placeholder whitening/colouring transform. It now uses `WCT2`/`NLUTNet` from the arch modules; WCT2 runs on the CPU because DirectML cannot execute its depthwise transposed convolutions, and NLUT loads the four `SB2..SB5` blocks its LUT classifier consumes, ignoring the checkpoint's unused `SB1`/`blurer` tensors. |
| Reencoding | `process()` read an undefined `params` variable (masked by a missing `ProcessingAborted` import), and step parameters other than the ones named in the constructor — plus the constructor values themselves — never reached the FFmpeg command, so codec/preset/CRF/scale/FPS choices were silently ignored. The encoder now follows the selected catalog model instead of being ignored, so the `handbrake` entry no longer produces FFmpeg output behind the user's back: it is marked as not implemented (`"arch": null`), is hidden from the Reencoding picker, is shown as *Implemented: No* on the Models page, and fails with an explicit message if requested directly through the API. |

The eight pipelines that raise `ProcessingAborted` (color, denoise, face,
interpolation, reencoding, sharpen, upscale, video) now import it, so aborting a
video job raises the intended error instead of a masking `NameError`.

Model files can be large. Keep downloaded checkpoints in the local `models/`
directory and ensure you have enough disk space before downloading them.

## Licensing

The model implementations and pretrained checkpoints may have different
licenses. Review the license and usage terms from each upstream project and
checkpoint provider before using, redistributing, or packaging model files.
Listing a model here does not grant rights to its code or weights.
