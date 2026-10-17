"""
Reencoding pipeline using FFmpeg.
Supports various codecs, presets, and quality settings.
"""
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

from ...processing.pipeline.base import EnhancementPipeline, PipelineConfig, ProcessingContext, ProcessingResult, InferenceError, UnsupportedFormatError, ProcessingAborted
from ...models.registry import get_registry, default_model

logger = logging.getLogger(__name__)


class ReencodingPipeline(EnhancementPipeline):
    """Reencoding pipeline using FFmpeg."""

    config = PipelineConfig(
        name="ReencodingPipeline",
        feature="Reencoding",
        description="Transcode video/audio with FFmpeg using various presets",
        supported_formats=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".ts"],
        supports_video=True,
        supports_image=False,
        requires_gpu=False,
        default_parameters={
            "engine": "ffmpeg",
            "video_codec": "libx264",
            "preset": "medium",
            "crf": 23,
            "audio_codec": "aac",
            "audio_bitrate": "128k",
        },
        parameter_schema={
            "type": "object",
            "properties": {
                "engine": {"type": "string", "enum": ["ffmpeg", "handbrake"], "default": "ffmpeg"},
                "video_codec": {"type": "string", "enum": ["libx264", "libx265", "libvpx-vp9", "av1_nvenc", "hevc_nvenc"], "default": "libx264"},
                "preset": {"type": "string", "enum": ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], "default": "medium"},
                "crf": {"type": "integer", "minimum": 0, "maximum": 51, "default": 23},
                "audio_codec": {"type": "string", "enum": ["aac", "opus", "mp3", "copy"], "default": "aac"},
                "audio_bitrate": {"type": "string", "default": "128k"},
                "scale": {"type": "string", "description": "Scale filter (e.g., '1920:1080', '-1:720')", "default": ""},
                "fps": {"type": "number", "description": "Target FPS", "default": 0},
            }
        }
    )

    # Preset configurations
    PRESETS = {
        "h264_fast": {
            "video_codec": "libx264",
            "preset": "fast",
            "crf": 23,
        },
        "h264_quality": {
            "video_codec": "libx264",
            "preset": "slow",
            "crf": 18,
        },
        "h265_fast": {
            "video_codec": "libx265",
            "preset": "fast",
            "crf": 25,
        },
        "h265_quality": {
            "video_codec": "libx265",
            "preset": "slow",
            "crf": 20,
        },
        "web_optimized": {
            "video_codec": "libx264",
            "preset": "medium",
            "crf": 23,
            "movflags": "+faststart",
        },
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = 'auto',
        engine: Optional[str] = None,
        video_codec: str = 'libx264',
        preset: str = 'medium',
        crf: int = 23,
        audio_codec: str = 'aac',
        audio_bitrate: str = '128k',
        parameters: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ):
        model_name = model_name or default_model("Reencoding")
        super().__init__(model_name, device, parameters=parameters, task_id=task_id)
        # The catalog exposes one entry per encoder, so the engine follows the
        # selected model unless the caller names one explicitly. Previously the
        # engine was ignored and every selection silently ran FFmpeg.
        self.engine = (engine or model_name or "ffmpeg").lower()
        self.video_codec = video_codec
        self.preset = preset
        self.crf = crf
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self._ffmpeg_path = self._find_ffmpeg()

    def _effective_params(self) -> Dict[str, Any]:
        """Merge catalog defaults, ctor values and step parameters.

        Step parameters bind both to explicit constructor arguments (``crf``,
        ``video_codec``, ...) and to ``self.parameters`` (keys the constructor
        does not name, such as ``scale`` or ``fps``), so all three sources have
        to be folded together to get the values actually used.
        """
        params: Dict[str, Any] = {
            **self.config.default_parameters,
            "video_codec": self.video_codec,
            "preset": self.preset,
            "crf": self.crf,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
            "engine": self.engine,
            **self.parameters,
        }

        # Apply preset if specified
        preset_name = params.get("preset_name")
        if preset_name and preset_name in self.PRESETS:
            preset_params = self.PRESETS[preset_name]
            for k, v in preset_params.items():
                if k not in params:  # Don't override explicit params
                    params[k] = v

        return params

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable."""
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        # Check common locations
        for path in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "C:\\ffmpeg\\bin\\ffmpeg.exe"]:
            if Path(path).exists():
                return path
        return "ffmpeg"  # Let system PATH handle it

    def load_model(self, model_path: Union[str, Path, Dict[str, Path]]) -> bool:
        """Verify the selected encoder is available."""
        if self.engine != 'ffmpeg':
            raise InferenceError(
                f"The '{self.engine}' encoder is not implemented in this build. "
                "Select the FFmpeg model for Reencoding."
            )
        try:
            result = subprocess.run(
                [self._ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"FFmpeg found: {result.stdout.split(chr(10))[0]}")
                self._loaded = True
                return True
            else:
                raise InferenceError("FFmpeg not working")
        except FileNotFoundError:
            raise InferenceError("FFmpeg not found. Please install FFmpeg.")
        except Exception as e:
            raise InferenceError(f"FFmpeg check failed: {e}")

    def _build_ffmpeg_cmd(self, context: ProcessingContext) -> List[str]:
        """Build FFmpeg command line."""
        params = self._effective_params()

        cmd = [self._ffmpeg_path, "-y"]  # -y = overwrite

        # Input
        cmd.extend(["-i", str(context.input_path)])

        # Video encoding
        cmd.extend(["-c:v", params.get("video_codec", self.video_codec)])

        if params.get("video_codec") in ("libx264", "libx265"):
            cmd.extend(["-preset", params.get("preset", self.preset)])
            cmd.extend(["-crf", str(params.get("crf", self.crf))])

            # Profile for compatibility
            if params.get("video_codec") == "libx264":
                cmd.extend(["-profile:v", "high", "-level", "4.1"])
            elif params.get("video_codec") == "libx265":
                cmd.extend(["-tag:v", "hvc1"])

        elif params.get("video_codec") in ("av1_nvenc", "hevc_nvenc"):
            # NVENC hardware encoding
            cmd.extend(["-preset", params.get("preset", "p4")])
            cmd.extend(["-cq", str(params.get("crf", 25))])

        # Audio encoding
        audio_codec = params.get("audio_codec", self.audio_codec)
        if audio_codec != "copy":
            cmd.extend(["-c:a", audio_codec])
            if audio_codec in ("aac", "opus", "mp3"):
                cmd.extend(["-b:a", params.get("audio_bitrate", self.audio_bitrate)])
        else:
            cmd.extend(["-c:a", "copy"])

        # Scale filter
        scale = params.get("scale", "")
        if scale:
            cmd.extend(["-vf", f"scale={scale}"])

        # FPS filter
        fps = params.get("fps", 0)
        if fps > 0:
            vf = cmd[cmd.index("-vf") + 1] if "-vf" in cmd else ""
            if vf:
                vf += f",fps={fps}"
            else:
                vf = f"fps={fps}"
            if "-vf" in cmd:
                cmd[cmd.index("-vf") + 1] = vf
            else:
                cmd.extend(["-vf", vf])

        # Movflags for web optimization
        if params.get("movflags"):
            cmd.extend(["-movflags", params["movflags"]])

        # Pixel format for compatibility
        cmd.extend(["-pix_fmt", "yuv420p"])

        # Output
        cmd.append(str(context.output_path))

        return cmd

    def _get_duration(self, input_path: Path) -> float:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Transcode video using FFmpeg."""
        if not self._loaded:
            raise InferenceError("FFmpeg not available. Call load_model() first.")

        path = context.input_path
        if path.suffix.lower() not in self.config.supported_formats:
            raise UnsupportedFormatError(f"Unsupported format: {path.suffix}")

        # Get duration for progress tracking
        duration = self._get_duration(path)

        # The values actually handed to FFmpeg (defaults + ctor args + step
        # parameters), reported back in the result metadata below.
        effective = self._effective_params()

        try:
            cmd = self._build_ffmpeg_cmd(context)
            logger.info(f"FFmpeg command: {' '.join(cmd)}")

            context.report_progress(5, "Starting transcode")

            # Run FFmpeg with progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )

            # Parse progress from stderr
            import re
            time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

            while True:
                if context.abort_flag:
                    process.terminate()
                    raise ProcessingAborted("Transcoding aborted")

                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    # Parse time for progress
                    match = time_pattern.search(line)
                    if match and duration > 0:
                        h, m, s = match.groups()
                        current_time = int(h) * 3600 + int(m) * 60 + float(s)
                        progress = min(95, int((current_time / duration) * 100))
                        context.report_progress(progress, f"Transcoding: {current_time:.1f}/{duration:.1f}s")

            return_code = process.wait()

            if return_code != 0:
                stderr = process.stderr.read()
                raise InferenceError(f"FFmpeg failed (code {return_code}): {stderr}")

            context.report_progress(100, "Transcoding complete")

            return ProcessingResult(
                success=True,
                input_path=context.input_path,
                output_path=context.output_path,
                metadata={
                    'engine': effective.get('engine', self.engine),
                    'video_codec': effective.get("video_codec", self.video_codec),
                    'preset': effective.get("preset", self.preset),
                    'crf': effective.get("crf", self.crf),
                    'audio_codec': effective.get("audio_codec", self.audio_codec),
                    'duration': duration,
                },
                model_used="FFmpeg",
                device_used="CPU",
            )

        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Transcoding failed: {e}")


from ...processing.pipeline.registry import register_pipeline

register_pipeline(
    feature="Reencoding",
    pipeline_class=ReencodingPipeline,
    available_models=get_registry().get_available_model_names("Reencoding"),
    default_model=get_registry().get_default_model("Reencoding"),
)