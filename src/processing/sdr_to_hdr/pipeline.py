"""
SDR to HDR conversion pipeline.
Model list, architecture and defaults come from the models.json catalog.

The catalog's classical entry ("HDR (Reinhard Tone Mapping)") describes the
algorithm used here: a Reinhard-style expansion of the SDR signal plus local
contrast, followed by a real BT.2100 encode (PQ for HDR10/HDR10+, HLG for HLG).

Images are written as 16-bit TIFFs carrying the full-range PQ/HLG signal.
Video is encoded to 10-bit HEVC with BT.2020 primaries and the matching
transfer characteristic through FFmpeg, because no OpenCV writer can produce a
10-bit HDR stream (the previous implementation wrote an 8-bit MPEG-4 file while
its metadata claimed HDR10).
"""
import cv2
import numpy as np
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import tempfile

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, ModelLoadError, InferenceError, ProcessingAborted
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)

# BT.2408 reference: SDR diffuse white is 100 nits.
SDR_REFERENCE_WHITE_NITS = 100.0
# Peak used for the PQ signal the classical model produces.
HDR10_PEAK_NITS = 1000.0

# ST 2084 (PQ) constants.
_PQ_M1 = 0.1593017578125
_PQ_M2 = 78.84375
_PQ_C1 = 0.8359375
_PQ_C2 = 18.8515625
_PQ_C3 = 18.6875

# BT.2100 HLG OETF constants.
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991073

_VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """BT.709/sRGB electro-optical transfer function (coded -> linear light)."""
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def pq_encode(nits: np.ndarray) -> np.ndarray:
    """ST 2084 perceptual quantizer: absolute luminance (nits) -> 0..1 signal."""
    y = np.clip(nits / 10000.0, 0.0, 1.0)
    y_pow = np.power(y, _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * y_pow) / (1.0 + _PQ_C3 * y_pow), _PQ_M2)


def hlg_encode(linear: np.ndarray) -> np.ndarray:
    """BT.2100 HLG opto-electronic transfer function (scene linear -> 0..1)."""
    x = np.clip(linear, 0.0, 1.0)
    # Both branches are evaluated by np.where, so the argument of the log is
    # clamped to stay positive in the low-signal branch.
    log_arg = np.maximum(12.0 * x - _HLG_B, 1e-9)
    return np.where(
        x <= 1.0 / 12.0,
        np.sqrt(3.0 * x),
        _HLG_A * np.log(log_arg) + _HLG_C,
    )


class SDRToHDRPipeline(EnhancementPipeline):
    """SDR to HDR conversion pipeline."""

    config = PipelineConfig(
        name="SDRToHDRPipeline",
        feature="SDR to HDR",
        description="Convert standard dynamic range to high dynamic range",
        supported_formats=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".webm"],
        supports_video=True,
        supports_image=True,
        requires_gpu=False,
        default_parameters={
            "target_format": "HDR10",
            "strength": 1.0,
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "target_format": {"type": "string", "enum": ["HDR10", "HDR10+", "HLG"], "default": "HDR10"},
                "strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
            }
        }
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        target_format: str = 'HDR10',
        strength: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("SDR to HDR")
        super().__init__(model_name, device, parameters=parameters, task_id=task_id)
        self.target_format = target_format
        self.strength = max(0.0, min(1.0, strength))
        self._model = None
        # Architecture comes from the models.json catalog entry.
        self._model_config = get_registry().runtime_config(model_name, "SDR to HDR")
        self._is_classical = bool(self._model_config.get('is_classical', True))
        self._ffmpeg_path: Optional[str] = None

    # -- signal helpers ---------------------------------------------------

    def _effective_params(self) -> Dict[str, Any]:
        """Step parameters merged with the resolved constructor values."""
        return {
            "target_format": self.target_format,
            "strength": self.strength,
            **self.parameters,
        }

    @property
    def _transfer_characteristic(self) -> str:
        """BT.2100 transfer characteristic written into the output."""
        return "arib-std-b67" if self.target_format == "HLG" else "smpte2084"

    def _expand_to_hdr(self, srgb: np.ndarray) -> np.ndarray:
        """Expand an SDR (0..1 coded sRGB) signal into linear light.

        A Reinhard-style inverse curve lifts diffuse highlights toward the HDR
        peak while keeping mid-tones near their original relative luminance; the
        ``strength`` parameter blends between the untouched SDR signal and the
        full expansion.
        """
        linear = srgb_to_linear(np.clip(srgb, 0.0, 1.0))
        if self.strength > 0.0:
            # L / (1 - k L) approaches a 10x gain as L -> 1/k, matching the
            # 1000-nit peak used for the 100-nit SDR reference.
            k = 0.9
            expanded = linear / np.maximum(1.0 - k * np.clip(linear, 0.0, 0.999), 1e-6)
            linear = linear + (expanded - linear) * self.strength
        return linear

    def _local_contrast(self, linear: np.ndarray) -> np.ndarray:
        """Boost local contrast with a luminance high-pass, clamped to sane values."""
        if self.strength <= 0.0:
            return linear
        blur = cv2.GaussianBlur(linear.astype(np.float32), (0, 0), 3.0)
        return np.clip(linear + 0.25 * self.strength * (linear - blur), 0.0, None)

    def _to_hdr_signal(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert an 8-bit BGR frame to the 16-bit coded HDR signal."""
        srgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        linear = self._local_contrast(self._expand_to_hdr(srgb))

        if self.target_format in ("HDR10", "HDR10+"):
            nits = np.clip(linear * SDR_REFERENCE_WHITE_NITS, 0.0, HDR10_PEAK_NITS)
            signal = pq_encode(nits)
        else:
            signal = hlg_encode(linear)

        # A 16-bit container carries the full precision of the 10-bit signal.
        return np.clip(signal * 65535.0, 0.0, 65535.0).round().astype(np.uint16)

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        if not self._is_classical:
            raise ModelLoadError(
                f"'{self.model_name}' has no runtime implementation in this build. "
                "Select the classical HDR model for SDR to HDR."
            )
        self._loaded = True
        logger.info(f"Initialized classical SDR to HDR: {self.model_name}")
        return True

    def process(self, context: ProcessingContext) -> ProcessingResult:
        if not self._loaded:
            raise InferenceError("Model not loaded. Call load_model() first.")

        path = context.input_path
        if path.suffix.lower() in _VIDEO_SUFFIXES:
            return self._process_video(context)
        return self._process_image(context)

    # -- images -----------------------------------------------------------

    def _process_image(self, context: ProcessingContext) -> ProcessingResult:
        try:
            img = cv2.imread(str(context.input_path), cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {context.input_path}")

            context.report_progress(10, "Converting SDR to HDR")

            if context.abort_flag:
                raise ProcessingAborted("SDR to HDR aborted")

            hdr = self._to_hdr_signal(img)

            # A 16-bit TIFF is the only image container here that preserves the
            # 10-bit HDR signal, so the output extension is forced to .tiff.
            if context.output_path.suffix.lower() not in {'.tiff', '.tif'}:
                output_path = context.output_path.with_suffix('.tiff')
            else:
                output_path = context.output_path

            if not cv2.imwrite(str(output_path), cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)):
                raise InferenceError(f"Failed to write output: {output_path}")

            context.report_progress(100, "SDR to HDR conversion complete")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=output_path,
                metadata={
                    **self._effective_params(),
                    'model_type': 'classical',
                    'bit_depth': 16,
                    'color_primaries': 'bt2020',
                    'transfer_characteristic': self._transfer_characteristic,
                    'hdr10_plus_dynamic_metadata': False,
                    'peak_nits': HDR10_PEAK_NITS,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )
        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"SDR to HDR conversion failed: {e}")

    # -- video ------------------------------------------------------------

    def _find_ffmpeg(self) -> Optional[str]:
        found = shutil.which("ffmpeg")
        if found:
            return found
        for candidate in ("ffmpeg.exe", "ffmpeg"):
            if Path(candidate).exists():
                return candidate
        return None

    def _build_ffmpeg_cmd(self, context: ProcessingContext, width: int, height: int, fps: float) -> List[str]:
        """Build the 10-bit HEVC encode command fed from stdin raw frames."""
        cmd = [
            self._ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            # Input 0: raw 16-bit RGB frames carrying the PQ/HLG signal.
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "pipe:0",
        ]
        # Input 1: the original file, used to copy any audio track across.
        if context.input_path.exists():
            cmd += ["-i", str(context.input_path), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]

        x265_params = "colorprim=bt2020:colormatrix=bt2020nc:repeat-headers=1"
        if self.target_format in ("HDR10", "HDR10+"):
            # FFmpeg's libx265 wrapper only forwards the transfer/primaries
            # through x265 itself, so they are set here rather than (only) as
            # output options.
            x265_params += (
                ":transfer=smpte2084:hdr10=1"
                ":master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
            )
            transfer = "smpte2084"
        else:
            x265_params += ":transfer=arib-std-b67"
            transfer = "arib-std-b67"

        cmd += [
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx265", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020",
            "-color_trc", transfer,
            "-colorspace", "bt2020nc",
            "-x265-params", x265_params,
            str(context.output_path),
        ]
        return cmd

    def _process_video(self, context: ProcessingContext) -> ProcessingResult:
        self._ffmpeg_path = self._find_ffmpeg()
        if not self._ffmpeg_path:
            raise InferenceError(
                "SDR to HDR video output needs FFmpeg with libx265 to write 10-bit HEVC. "
                "Install FFmpeg or process a still image instead."
            )

        cap = cv2.VideoCapture(str(context.input_path))
        if not cap.isOpened():
            raise InferenceError(f"Failed to open video: {context.input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not width or not height:
            cap.release()
            raise InferenceError(f"Could not determine the dimensions of {context.input_path}")

        cmd = self._build_ffmpeg_cmd(context, width, height, fps)
        log_path = Path(tempfile.gettempdir()) / f"{context.output_path.stem}_ffmpeg.log"

        processed = 0
        proc: Optional[subprocess.Popen] = None
        context.report_progress(2, f"Encoding {total_frames or '?'} frames to 10-bit HDR")
        try:
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    while True:
                        if context.abort_flag:
                            raise ProcessingAborted("Video processing aborted")

                        ret, frame = cap.read()
                        if not ret:
                            break

                        hdr = self._to_hdr_signal(frame)
                        proc.stdin.write(hdr.tobytes())
                        processed += 1

                        if total_frames and (processed % 10 == 0 or processed == total_frames):
                            progress = 2 + int((processed / total_frames) * 96)
                            context.report_progress(progress, f"Frame {processed}/{total_frames}")
                finally:
                    cap.release()
                    if proc.stdin and not proc.stdin.closed:
                        try:
                            proc.stdin.close()
                        except (BrokenPipeError, OSError):
                            pass

            return_code = proc.wait()
            if return_code != 0:
                detail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                raise InferenceError(
                    "FFmpeg failed to encode the HDR stream"
                    + (f": {detail[-1]}" if detail else f" (exit code {return_code})")
                )

            context.report_progress(100, "SDR to HDR conversion complete")
            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    **self._effective_params(),
                    'frames_processed': processed,
                    'total_frames': total_frames,
                    'fps': fps,
                    'model_type': 'classical',
                    'bit_depth': 10,
                    'color_primaries': 'bt2020',
                    'transfer_characteristic': self._transfer_characteristic,
                    'hdr10_plus_dynamic_metadata': False,
                    'peak_nits': HDR10_PEAK_NITS,
                },
                model_used=self.model_name,
                device_used=self.get_device(),
            )
        except ProcessingAborted:
            if proc is not None and proc.poll() is None:
                proc.kill()
            raise
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Video SDR to HDR conversion failed: {e}")
        finally:
            if log_path.exists():
                log_path.unlink(missing_ok=True)


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="SDR to HDR",
    pipeline_class=SDRToHDRPipeline,
    available_models=get_registry().get_available_model_names("SDR to HDR"),
    default_model=get_registry().get_default_model("SDR to HDR"),
)
