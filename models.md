# Model catalog

openVideoAI keeps metadata for open-source image and video restoration models in
[models.json](models.json). The catalog describes potential model integrations;
it is not a claim that every model is already runnable in the application.

It currently lists **28 unique models**: **18** have a catalog download URL and
**10** do not. The model manager skips repeated names in the catalog.

> **Current status:** the app can list model metadata, manage model downloads,
> and run the Real-ESRGAN upscaling workflow plus NAFNet image denoising and
> deblurring workflows. The other catalog entries are model metadata and/or
> downloadable assets; they do not necessarily have an inference integration
> in the app. Treat listed sizes as
> estimates, not verified benchmarks.

## Catalog models

| Model | Main use | Media | Approx. catalog size | Type | Catalog hardware |
| --- | --- | --- | --- | --- | --- |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | General-purpose upscaling and texture restoration | Images | ~65 MB | AI | CUDA |
| [CodeFormer](https://github.com/sczhou/CodeFormer) | Face restoration | Images | ~150 MB | AI | CUDA |
| [NAFNet-SIDD](https://huggingface.co/nyanko7/nafnet-models/blob/main/NAFNet-SIDD-width64.pth) | Sensor-noise removal | Images | ~464 MB | AI | CPU-capable |
| [NAFNet-REDS](https://huggingface.co/nyanko7/nafnet-models/blob/main/NAFNet-REDS-width64.pth) | General video restoration | Video | ~272 MB | AI | CPU-capable |
| [NAFNet-GoPro](https://huggingface.co/nyanko7/nafnet-models/blob/main/NAFNet-GoPro-width64.pth) | Motion deblurring | Images / video | ~272 MB | AI | CPU-capable |
| [SCUNet](https://github.com/cszn/SCUNet) | Denoising and compression-artifact cleanup | Images | ~45 MB | AI | CUDA |
| [RIFE](https://github.com/vladmandic/rife) | Frame interpolation and slow motion | Video | ~20 MB | AI | CPU-capable |
| [RIFE v4.26 (PyTorch)](https://huggingface.co/thornmaze/RIFE) | RIFE v4.26 checkpoint archive | Video | ~23 MB | AI | CUDA |
| [RIFE 4.25 (MLX)](https://huggingface.co/mlx-community/RIFE-4.25) | Apple Silicon MLX frame interpolation | Video | ~23 MB | AI | Apple Silicon / Metal |
| [WMS-RIFE v3 (PyTorch)](https://huggingface.co/Anson-Saju-George/wms-rifev3) | Weather and satellite imagery frame interpolation | Video | ~12 MB | AI | CUDA |
| [RIFE (PyTorch Safetensors)](https://huggingface.co/TensorForger/RIFE-safetensors) | RIFE v4 PyTorch frame interpolation | Video | ~12 MB | AI | CUDA |
| [RIFE v4.9 (ONNX)](https://huggingface.co/edgetools/rife) | ONNX frame interpolation | Video | ~21 MB | AI | CUDA |
| [FILM](https://github.com/google-research/frame-interpolation) | Frame interpolation with large motion | Video / frame sequences | ~130 MB | AI | CUDA |
| [Farnebäck Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) | Polynomial-expansion dense optical flow | Video | Built-in (&lt; 1 MB) | Non-AI (math) | CPU |
| [DIS Optical Flow](https://docs.opencv.org/4.x/da/d06/classcv_1_1DISOpticalFlow.html) | Dense Inverse Search optical flow | Video | Built-in (&lt; 1 MB) | Non-AI (math) | CPU |
| [Dual TV-L1 Optical Flow](https://docs.opencv.org/4.x/dc/d47/classcv_1_1DualTVL1OpticalFlow.html) | Total-variation energy-minimization optical flow | Video | Built-in (&lt; 1 MB) | Non-AI (math) | CPU |
| [Lucas-Kanade (Pyramidal LK)](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) | Classical pyramidal differential optical flow | Video | Built-in (&lt; 1 MB) | Non-AI (math) | CPU |
| [RAFT Optical Flow](https://github.com/princeton-vl/RAFT) | Recurrent deep-learning optical flow | Video | ~20 MB | AI | CUDA |
| [FlowFormer Optical Flow](https://github.com/drinkingcoder/FlowFormer) | Transformer-based optical flow | Video | ~40 MB | AI | CUDA |
| [GMA Optical Flow](https://github.com/zacjiang/GMA) | Global motion aggregation optical flow | Video | ~25 MB | AI | CUDA |
| [SPyNet Optical Flow](https://github.com/princeton-vl/RAFT) | Lightweight spatial pyramid optical flow | Video | ~5 MB | AI | CPU-capable |
| [EpicFlow Optical Flow](https://github.com/reuter/epicflow) | Edge-preserving correspondence interpolation | Video | Built-in (&lt; 1 MB) | Non-AI (math) | CPU |
| [Sapsan-VFI](https://huggingface.co/SnJake/Sapsan-VFI) | Frame interpolation and slow motion | Video | ~270 MB | AI | CUDA |
| [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) | Multi-frame video restoration, including a catalog feature for LOG restoration | Video | ~120 MB | AI | CUDA |
| [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | Anime and 2D image upscaling and cleanup | Images | ~40 MB | AI | CUDA |
| [NLUT](https://github.com/semchan/NLUT) | Reference-based neural color matching | Images / video color-grading metadata | ~45 MB | AI | CUDA |
| [Deep Color Transfer](https://github.com/codeslake/Color_Transfer_Histogram_Analogy) | Semantic color transfer from a reference | Images | ~110 MB | AI | CUDA |
| [Reinhard Transfer](https://github.com/jrosebr1/color_transfer) | Lightweight statistical color matching | Images | &lt; 1 MB | Non-AI (math) | CPU |

The hardware column reflects the catalog's declared requirements and CPU
fallback, not a guarantee that the app currently implements that runtime or
can run every listed model on that device. See
[Downloads and model files](#downloads-and-model-files) before downloading.

The catalog also defines features for **Upscale**, **Face Recovery**,
**Sharpen / Deblur**, **Denoise**, **Frame Interpolation**, **SloMo**,
**Frame Recovery**, **Multipurpose Restoration**, **Correct Distortion**,
**Low-Light Enhancement**, **Colorization**, **Object Removal**, **Match
Color**, **Log Restoration**, **SDR to HDR**, and **Focus Fix**. These labels
describe intended use and catalog/UI availability; they do not imply that the
corresponding inference implementations are complete.

## Choosing a model

- For general image upscaling, start with Real-ESRGAN.
- For faces, consider CodeFormer.
- For sensor noise, consider NAFNet-SIDD; for motion blur, consider
  NAFNet-GoPro. NAFNet-REDS targets general video restoration.
- For anime or line art, consider Real-CUGAN.
- For generating intermediate video frames, consider RIFE or FILM.
- The catalog includes the non-AI optical-flow entries Farnebäck, DIS, Dual
  TV-L1, Lucas-Kanade (Pyramidal LK), and EpicFlow. They are marked as
  built-in, CPU-capable algorithms and do not need downloaded model weights;
  this catalog status does not mean they are already wired into the app's
  interpolation pipeline.
- The catalog also lists the deep-learning optical-flow entries RAFT,
  FlowFormer, GMA, and SPyNet. Their hardware requirements are metadata; the
  entries do not currently provide app-level optical-flow inference.
- The catalog includes RIFE checkpoints in several formats and variants:
  Practical-RIFE v4.26 as a ZIP archive, RIFE 4.25 for Apple's MLX runtime, a
  weather/satellite-specialized PyTorch v3 checkpoint, PyTorch weights in
  safetensors format, and RIFE v4.9 in ONNX format. Each needs compatible model
  code and a matching runtime; downloading a checkpoint alone does not add
  interpolation to the app.
- For video restoration, consider RealBasicVSR.
- For reference-based color matching, the catalog lists NLUT, Deep Color
  Transfer, and Reinhard Transfer. Their entries are metadata and do not by
  themselves provide a runnable integration.

## LOG footage module

The catalog contains a **LOG Footage Enhancer & Neural Color Grade** module
configuration for footage such as S-Log, C-Log, V-Log, LogC, and D-Log. Its
proposed stages are temporal denoising, exposure and contrast recovery, and
neural tone mapping/color restoration. The UI exposes the configured input
profile, denoising strength, shadow recovery, and output color-space settings.

This is configuration-only at present. The UI does not execute the listed
models or produce a processed Rec.709/HDR export, and the values it collects
should not be treated as an implemented color-management pipeline.

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

The catalog is the single source of truth for these requirements:

- CUDA-required models are only downloadable (and visible as compatible) on
  machines with a detected CUDA-capable GPU.
- Metal-required models are only downloadable on Apple Silicon devices.
- Non-AI entries with no download URL are treated as **built-in**: they are
  reported as already available, need no download, and appear in the UI filter
  "Non-AI (Pure Math)".

Model files can be large. Keep downloaded checkpoints in the local `models/`
directory and ensure you have enough disk space before downloading them.

## Licensing

The model implementations and pretrained checkpoints may have different
licenses. Review the license and usage terms from each upstream project and
checkpoint provider before using, redistributing, or packaging model files.
Listing a model here does not grant rights to its code or weights.
