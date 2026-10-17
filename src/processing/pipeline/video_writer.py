"""
Video output sinks.

Every pipeline that writes a video used to open ``cv2.VideoWriter`` with the
``mp4v`` fourcc, which ignored the codec, the quality, the container and - most
visibly - dropped the audio track entirely.

``open_video_sink`` keeps that behaviour for the throwaway intermediate files a
pipeline chain produces (they are re-read immediately) but hands the *final*
output to FFmpeg, so the export dialog's codec, profile, bitrate, quality,
container, audio mode and LUT actually reach the file. When the final output is
an image sequence the frames are written as numbered image files instead.

The rules about which combination is legal live in ``services.export_service``;
this module turns a normalised option set into FFmpeg arguments.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from .base import InferenceError, ProcessingContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFmpeg discovery
# ---------------------------------------------------------------------------
def ffmpeg_path() -> Optional[str]:
    """The FFmpeg executable, or ``None`` when this machine has none."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe"):
        if Path(candidate).exists():
            return candidate
    return None


def ffprobe_path() -> Optional[str]:
    """The FFprobe executable that sits next to FFmpeg."""
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = ffmpeg_path()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
        if sibling.exists():
            return str(sibling)
    return None


# ---------------------------------------------------------------------------
# Encoder arguments
# ---------------------------------------------------------------------------
# Encoding speed follows the quality level: a "high" export is worth more time.
_X264_X265_PRESETS = {"low": "veryfast", "medium": "medium", "high": "slow"}
_VP9_CPU_USED = {"low": 4, "medium": 2, "high": 0}
_NVENC_PRESETS = {"low": "p4", "medium": "p5", "high": "p7"}

# Suffixes the built-in OpenCV writer can actually mux.
_CV2_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class EncodeSettings:
    """A normalised export option set, as consumed by the encoders."""

    output_type: str = "video"
    container: str = "mp4"
    codec: str = "libx264"
    profile: str = ""
    quality: str = "high"
    crf: int = 18
    bitrate_mode: str = "dynamic"
    bitrate_mbps: float = 0.0
    audio_mode: str = "none"
    audio_codec: Optional[str] = None
    audio_bitrate: str = ""
    lut_path: str = ""
    image_format: str = "png"
    jpg_quality: int = 95
    audio_source: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def wants_audio(self) -> bool:
        return self.audio_mode not in ("", "none", None)

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Optional["EncodeSettings"]:
        raw = parameters.get("export")
        if not isinstance(raw, dict):
            return None
        source = raw.get("audio_source")
        return cls(
            output_type=str(raw.get("output_type") or "video"),
            container=str(raw.get("container") or "mp4"),
            codec=str(raw.get("codec") or "libx264"),
            profile=str(raw.get("profile") or ""),
            quality=str(raw.get("quality") or "high"),
            crf=int(raw.get("crf") or 18),
            bitrate_mode=str(raw.get("bitrate_mode") or "dynamic"),
            bitrate_mbps=float(raw.get("bitrate_mbps") or 0.0),
            audio_mode=str(raw.get("audio_mode") or "none"),
            audio_codec=raw.get("audio_codec"),
            audio_bitrate=str(raw.get("audio_bitrate") or ""),
            lut_path=str(raw.get("lut_path") or ""),
            image_format=str(raw.get("image_format") or "png"),
            jpg_quality=int(raw.get("jpg_quality") or 95),
            audio_source=Path(source) if source else None,
        )


def escape_filter_path(path: str) -> str:
    """Escape a filesystem path for use inside an FFmpeg filter argument."""
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def build_filter_chain(settings: EncodeSettings, width: int, height: int) -> List[str]:
    """Filters the encoded video needs (LUT, even-dimension guard)."""
    filters: List[str] = []
    if settings.lut_path:
        filters.append(f"lut3d=file={escape_filter_path(settings.lut_path)}")
    # yuv420p (and most encoders) need even dimensions.
    if width % 2 or height % 2:
        filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    return filters


def build_video_args(settings: EncodeSettings, *, width: int = 0, height: int = 0) -> List[str]:
    """FFmpeg output arguments for the chosen video encoder."""
    codec = settings.codec or "libx264"
    quality = settings.quality if settings.quality in _X264_X265_PRESETS else "medium"
    constant = settings.bitrate_mode == "constant" and settings.bitrate_mbps > 0

    args: List[str] = ["-c:v", codec]

    if codec in ("libx264", "libx265"):
        args += ["-preset", _X264_X265_PRESETS.get(quality, "medium")]
        if constant:
            args += [
                "-b:v", f"{settings.bitrate_mbps:g}M",
                "-maxrate", f"{settings.bitrate_mbps * 1.5:g}M",
                "-bufsize", f"{settings.bitrate_mbps * 3:g}M",
            ]
        else:
            args += ["-crf", str(settings.crf)]
        if settings.profile:
            args += ["-profile:v", settings.profile]
        if codec == "libx265":
            # Apple players only accept HEVC in MP4 with the hvc1 tag.
            args += ["-tag:v", "hvc1"]

    elif codec == "libvpx-vp9":
        args += ["-deadline", "good", "-cpu-used", str(_VP9_CPU_USED.get(quality, 2))]
        if constant:
            args += ["-b:v", f"{settings.bitrate_mbps:g}M"]
        else:
            args += ["-crf", str(settings.crf), "-b:v", "0"]

    elif codec in ("av1_nvenc", "hevc_nvenc"):
        args += ["-preset", _NVENC_PRESETS.get(quality, "p5")]
        if constant:
            args += ["-b:v", f"{settings.bitrate_mbps:g}M"]
        else:
            # NVENC's rate control takes a target quality instead of a CRF.
            args += ["-rc", "vbr", "-cq", str(settings.crf), "-b:v", "0"]
        if settings.profile and codec == "hevc_nvenc":
            args += ["-profile:v", settings.profile]

    pix_fmt = "yuv420p10le" if settings.profile == "main10" else "yuv420p"
    args += ["-pix_fmt", pix_fmt]

    filters = build_filter_chain(settings, width, height)
    if filters:
        args += ["-vf", ",".join(filters)]

    if settings.container in ("mp4", "mov"):
        # Start playing before the whole file has arrived.
        args += ["-movflags", "+faststart"]

    return args


def build_audio_args(settings: EncodeSettings) -> List[str]:
    """FFmpeg output arguments for the audio track."""
    if not settings.wants_audio:
        return ["-an"]
    codec = settings.audio_codec or "aac"
    args = ["-c:a", codec]
    if codec != "copy" and settings.audio_bitrate:
        args += ["-b:a", settings.audio_bitrate]
    return args


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
class VideoSink:
    """Writes frames to the configured output."""

    path: Path
    frames_written: int = 0
    warnings: List[str]
    encoder: str = "builtin"

    def write(self, frame) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class FFmpegSink(VideoSink):
    """Pipes raw frames to FFmpeg, which encodes, applies the LUT and muxes audio."""

    def __init__(self, path: Path, fps: float, size: Tuple[int, int], settings: EncodeSettings):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_written = 0
        self.warnings = list(settings.warnings)
        self.encoder = settings.codec

        executable = ffmpeg_path()
        if not executable:  # pragma: no cover - guarded by the caller
            raise InferenceError("FFmpeg was not found")

        width, height = int(size[0]), int(size[1])
        self._size = (width, height)
        # Frames of a different size are resized rather than failing the export.
        self._expects_size = (width, height)

        command = [
            executable, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{_safe_fps(fps)}",
            "-i", "pipe:0",
        ]

        use_audio = settings.wants_audio and settings.audio_source is not None and Path(settings.audio_source).exists()
        if use_audio:
            command += ["-i", str(settings.audio_source)]
        if settings.wants_audio and not use_audio:
            self.warnings.append("The source has no audio track; exporting video only.")

        command += ["-map", "0:v:0"]
        if use_audio:
            # "?" keeps the mapping optional, and the audio is taken from the
            # original file because the intermediate stages are silent.
            command += ["-map", "1:a:0?"]

        command += build_video_args(settings, width=width, height=height)
        if use_audio:
            command += build_audio_args(settings)
        else:
            command += ["-an"]
        command.append(str(self.path))

        logger.info("FFmpeg export: %s", " ".join(command))
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except OSError as error:  # pragma: no cover - FFmpeg vanished
            raise InferenceError(f"Could not start FFmpeg: {error}")

    def write(self, frame) -> None:
        if frame is None:
            return
        height, width = frame.shape[:2]
        if (width, height) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_CUBIC)
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            # FFmpeg died; close() reports why.
            self._broken = True
        self.frames_written += 1

    def close(self) -> None:
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        return_code = self._process.wait()

        self._stderr.seek(0)
        message = self._stderr.read().decode("utf-8", "replace").strip()
        self._stderr.close()

        if return_code != 0:
            self.path.unlink(missing_ok=True)
            tail = message.splitlines()[-4:] if message else []
            raise InferenceError(
                f"FFmpeg could not write {self.path.name}"
                + (f": {' | '.join(tail)}" if tail else f" (exit code {return_code})")
            )


class Cv2Sink(VideoSink):
    """The built-in MPEG-4 writer, used for intermediates and without FFmpeg."""

    encoder = "builtin (mp4v)"

    def __init__(self, path: Path, fps: float, size: Tuple[int, int]):
        self.warnings = []
        self.frames_written = 0
        requested = Path(path)
        self.path = self._mp4_path(requested)
        if self.path != requested:
            self.warnings.append(
                f"The built-in writer only produces MP4; writing {self.path.name}."
            )
            requested = self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, _safe_fps(fps), (int(size[0]), int(size[1]))
        )
        if not self._writer.isOpened():
            raise InferenceError(f"Failed to create output video: {self.path}")

    @staticmethod
    def _mp4_path(path: Path) -> Path:
        if path.suffix.lower() in _CV2_SUFFIXES:
            return path
        return path.with_suffix(".mp4")

    def write(self, frame) -> None:
        if frame is None:
            return
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        self._writer.release()


class ImageSequenceSink(VideoSink):
    """Writes one image file per frame into a folder."""

    encoder = "image sequence"

    def __init__(self, path: Path, settings: EncodeSettings, warnings: Optional[List[str]] = None):
        self.warnings = list(warnings or [])
        self.frames_written = 0
        planned = Path(path)
        # The caller plans a file path (``<name>.<format>``); the frames go into
        # a folder next to it so the service can move the folder into place.
        self.path = planned.parent / planned.stem
        self.path.mkdir(parents=True, exist_ok=True)
        self._suffix = f".{settings.image_format or 'png'}"

        self._params: List[int] = []
        if self._suffix in (".jpg", ".jpeg"):
            self._params = [cv2.IMWRITE_JPEG_QUALITY, int(settings.jpg_quality)]
        elif self._suffix == ".webp":
            self._params = [cv2.IMWRITE_WEBP_QUALITY, int(settings.jpg_quality)]
        elif self._suffix == ".png":
            self._params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    def write(self, frame) -> None:
        if frame is None:
            return
        index = self.frames_written + 1
        target = self.path / f"{self.path.name}_{index:06d}{self._suffix}"
        if not cv2.imwrite(str(target), frame, self._params):
            raise InferenceError(f"Failed to write frame {index} to {target}")
        self.frames_written += 1

    def close(self) -> None:
        return


def open_video_sink(
    output_path: Path,
    *,
    fps: float,
    size: Tuple[int, int],
    context: ProcessingContext,
    final: Optional[bool] = None,
) -> VideoSink:
    """Open the right sink for this stage.

    The final stage of an export uses the dialog's settings; every other stage
    keeps the fast, self-contained MP4 writer because its file is only read by
    the next stage.
    """
    settings = EncodeSettings.from_parameters(context.parameters or {})
    is_final = context.parameters.get("export_final", True) if final is None else final

    warnings: List[str] = context.metadata.setdefault("export_warnings", [])

    if settings is not None and is_final:
        if settings.output_type == "image_sequence":
            sink = ImageSequenceSink(output_path, settings)
        elif ffmpeg_path():
            sink = FFmpegSink(output_path, fps, size, settings)
        else:
            sink = Cv2Sink(output_path, fps, size)
            sink.warnings.append(
                "FFmpeg was not found, so the built-in MP4 writer was used "
                "(MPEG-4 part 2, no audio)."
            )
    else:
        sink = Cv2Sink(output_path, fps, size)

    warnings.extend(sink.warnings)
    context.metadata["export_encoder"] = sink.encoder
    # The sink may have adjusted the path (an MP4-only fallback, or a frame
    # folder), so the pipeline reports what was really written.
    context.output_path = Path(sink.path)
    return sink


def _safe_fps(fps: float) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError):
        return 25.0
    if value <= 0 or value != value:  # 0 or NaN
        return 25.0
    return round(value, 6)


def probe_video(path: Path) -> Dict[str, Any]:
    """Width, height, fps and duration of a video (empty dict when unknown)."""
    probe = ffprobe_path()
    if not probe:
        return {}
    try:
        result = subprocess.run(
            [
                probe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        import json

        streams = json.loads(result.stdout or "{}").get("streams") or []
        if not streams:
            return {}
        stream = streams[0]
        rate = str(stream.get("r_frame_rate") or "0/1")
        numerator, _, denominator = rate.partition("/")
        fps = float(numerator) / float(denominator or 1) if float(denominator or 1) else 0.0
        return {
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "fps": fps,
            "duration": float(stream.get("duration") or 0.0),
        }
    except Exception:  # pragma: no cover - probing is best effort
        logger.debug("ffprobe failed for %s", path, exc_info=True)
        return {}
