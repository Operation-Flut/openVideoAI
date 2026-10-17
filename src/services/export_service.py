"""
Export options - the single source of truth for the Export dialog.

``export_form_schema()`` returns the field list the dialog renders, so the
encoder list, the containers, the audio modes, the quality levels, the wording
and the dependencies between fields all live here instead of in the UI.

``normalise_options()`` runs on every request as well, so a stale dialog (or a
hand-written API call) can never ask FFmpeg for a combination the muxer would
reject - an unsupported pick is corrected and reported as a warning rather than
silently producing a broken file.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.paths import exports_dir, luts_dir
# FFmpeg discovery is shared with the encoder that actually runs the command.
from ..processing.pipeline.video_writer import ffmpeg_path, ffprobe_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Codec:
    """One FFmpeg video encoder the export dialog can offer."""

    encoder: str
    label: str
    containers: Tuple[str, ...]
    # Quality level -> the value handed to the encoder (CRF for x264/x265/VP9,
    # CQ for the NVENC encoders). Levels are the UI's quality knob.
    quality: Tuple[Tuple[str, int], ...] = ()
    profiles: Tuple[str, ...] = ()
    help: str = ""
    # NVENC encoders have no software fallback; they need an NVIDIA GPU.
    requires_nvidia: bool = False


CODECS: Tuple[Codec, ...] = (
    Codec(
        encoder="libx264",
        label="H.264 (x264)",
        containers=("mp4", "mkv", "mov"),
        quality=(("low", 28), ("medium", 23), ("high", 18)),
        profiles=("baseline", "main", "high"),
        help="Plays everywhere. The safe default.",
    ),
    Codec(
        encoder="libx265",
        label="H.265 / HEVC (x265)",
        containers=("mp4", "mkv", "mov"),
        quality=(("low", 30), ("medium", 25), ("high", 20)),
        profiles=("main", "main10"),
        help="Roughly half the size of H.264 at the same quality; slower to encode.",
    ),
    Codec(
        encoder="libvpx-vp9",
        label="VP9 (libvpx)",
        containers=("webm", "mkv"),
        quality=(("low", 40), ("medium", 33), ("high", 26)),
        help="Royalty-free and made for the web; WebM only.",
    ),
    Codec(
        encoder="av1_nvenc",
        label="AV1 (NVIDIA NVENC)",
        containers=("mp4", "mkv"),
        quality=(("low", 34), ("medium", 30), ("high", 25)),
        requires_nvidia=True,
        help="Fast AV1, encoded on an NVIDIA GPU.",
    ),
    Codec(
        encoder="hevc_nvenc",
        label="H.265 / HEVC (NVIDIA NVENC)",
        containers=("mp4", "mkv", "mov"),
        quality=(("low", 30), ("medium", 26), ("high", 21)),
        requires_nvidia=True,
        help="Fast H.265, encoded on an NVIDIA GPU.",
    ),
)

CODECS_BY_ENCODER: Dict[str, Codec] = {codec.encoder: codec for codec in CODECS}

CONTAINERS: Tuple[Tuple[str, str, str], ...] = (
    ("mp4", "MP4", "Plays on phones, browsers and TVs."),
    ("mkv", "Matroska (MKV)", "Holds any codec; good for archiving."),
    ("mov", "QuickTime (MOV)", "Preferred by editors on macOS."),
    ("webm", "WebM", "For the web: VP9 video with Opus audio."),
)
CONTAINER_LABELS = {value: label for value, label, _ in CONTAINERS}
CONTAINER_HELP = {value: help_text for value, _, help_text in CONTAINERS}

# Which audio the container can carry. "copy" keeps the source track untouched.
AUDIO_MODES: Tuple[Dict[str, Any], ...] = (
    {
        "value": "copy",
        "label": "Copy from source",
        "codec": "copy",
        "help": "Keeps the original track as it is.",
    },
    {
        "value": "aac",
        "label": "AAC (192 kbps)",
        "codec": "aac",
        "bitrate": "192k",
        "help": "Re-encodes the audio; plays everywhere.",
    },
    {
        "value": "opus",
        "label": "Opus (160 kbps)",
        "codec": "libopus",
        "bitrate": "160k",
        "help": "Small and high quality; for WebM/MKV.",
    },
    {
        "value": "mp3",
        "label": "MP3 (192 kbps)",
        "codec": "libmp3lame",
        "bitrate": "192k",
        "help": "Older players and MP4/MKV files.",
    },
    {"value": "none", "label": "No audio", "codec": None, "help": "Video only."},
)
AUDIO_MODES_BY_VALUE = {mode["value"]: mode for mode in AUDIO_MODES}

# Container -> the audio modes it accepts, first entry is the fallback.
AUDIO_MODES_BY_CONTAINER: Dict[str, Tuple[str, ...]] = {
    "mp4": ("copy", "aac", "mp3", "none"),
    "mkv": ("copy", "aac", "opus", "mp3", "none"),
    "mov": ("copy", "aac", "none"),
    "webm": ("opus", "none"),
}

# Which encoded audio codecs each container accepts (used to sanity-check
# "copy": an Opus track cannot be copied into an MP4).
CONTAINER_AUDIO_CODECS: Dict[str, Tuple[str, ...]] = {
    "mp4": ("aac", "mp3", "ac3", "eac3", "mp2", "alac"),
    "mkv": ("aac", "mp3", "opus", "vorbis", "ac3", "eac3", "flac", "alac", "pcm_s16le"),
    "mov": ("aac", "mp3", "alac", "pcm_s16le"),
    "webm": ("opus", "vorbis"),
}

QUALITY_LEVELS: Tuple[Tuple[str, str, str], ...] = (
    ("low", "Low", "Smallest file, quickest encode."),
    ("medium", "Medium", "Balanced quality and size."),
    ("high", "High", "Best quality; biggest file and slowest encode."),
)

BITRATE_MODES: Tuple[Tuple[str, str, str], ...] = (
    ("dynamic", "Dynamic", "One pass aiming at a quality level; the best quality per byte."),
    ("constant", "Constant", "A fixed bitrate, so the file size is predictable."),
)

IMAGE_FORMATS: Tuple[Tuple[str, str], ...] = (
    ("png", "PNG (lossless)"),
    ("jpg", "JPEG"),
    ("webp", "WebP"),
)

OUTPUT_TYPES: Tuple[Tuple[str, str, str], ...] = (
    ("video", "Video", "One encoded video file."),
    ("image_sequence", "Image sequence", "One file per frame, in a folder."),
)

SAVE_TARGETS: Tuple[Tuple[str, str, str], ...] = (
    ("original", "Original folder", "Next to the file being processed."),
    ("exports", "Exports folder", "openVideoAI's own exports folder."),
    ("custom", "Choose a folder", "A folder of your choice."),
)

CONSTANT_BITRATE_MIN_MBPS = 1.0
CONSTANT_BITRATE_MAX_MBPS = 200.0
CONSTANT_BITRATE_DEFAULT_MBPS = 12.0

# Codecs the built-in (OpenCV) writer can mux when FFmpeg is missing. It only
# writes MPEG-4 part 2, so the container is forced to MP4.
FALLBACK_CODEC = "mp4v"
FALLBACK_CONTAINER = "mp4"

_LUT_SUFFIXES = (".cube",)


# ---------------------------------------------------------------------------
# Machine capabilities
# ---------------------------------------------------------------------------
def has_nvidia_encoder() -> bool:
    """Whether an NVIDIA GPU (i.e. NVENC) is available on this machine."""
    try:
        from ..utils.system_resources import get_platform_capabilities

        return bool(get_platform_capabilities().get("has_cuda"))
    except Exception:  # pragma: no cover - detection must never break the UI
        logger.debug("Could not detect CUDA for NVENC availability", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Naming and output location
# ---------------------------------------------------------------------------
def render_output_name(
    pattern: str,
    *,
    name: str,
    scale: Optional[int] = None,
) -> str:
    """Turn the naming pattern (Settings -> Export) into a concrete file name.

    ``{name}`` is the input's stem and ``{scale}`` the upscale factor. Unknown
    placeholders are left alone rather than crashing the export.
    """
    rendered = pattern or "{name}_enhanced"
    rendered = rendered.replace("{name}", name)
    rendered = rendered.replace("{scale}", str(scale if scale else 1))
    # A pattern may end up with characters Windows refuses in a file name.
    for bad in '<>:"/\\|?*':
        rendered = rendered.replace(bad, "_")
    return rendered.strip() or f"{name}_enhanced"


def strip_media_suffix(name: str) -> str:
    """Drop a trailing media extension so the container/format decides it."""
    known = {
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts",
        ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp",
    }
    suffix = Path(name).suffix.lower()
    if suffix in known:
        return Path(name).stem
    return name


def output_directory(save_to: str, export_dir_setting: str) -> Path:
    """The folder an export lands in for the chosen save target."""
    if save_to == "original":
        return Path()  # resolved against the input's folder by the caller
    if save_to == "custom":
        return Path(export_dir_setting or ".").expanduser()
    return exports_dir()


def output_extension(options: Dict[str, Any]) -> str:
    """The suffix the chosen options produce (image sequences included)."""
    if options.get("output_type") == "image_sequence":
        return f".{options.get('image_format', 'png')}"
    return f".{options.get('container', FALLBACK_CONTAINER)}"


def resolve_output_path(
    input_path: Path,
    options: Dict[str, Any],
    *,
    naming_pattern: str,
    scale: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Where an export should be written, honouring the dialog's choices.

    ``save_to == "original"`` keeps the output next to the source file; the
    other targets use the exports folder or a folder the user picked. With
    "overwrite existing" off an existing name gets a numeric suffix instead of
    being clobbered.
    """
    stem = options.get("filename")
    if not stem:
        stem = render_output_name(naming_pattern, name=input_path.stem, scale=scale)
    stem = strip_media_suffix(str(stem))

    suffix = output_extension(options)

    directory = output_directory(str(options.get("save_to") or "exports"), str(options.get("export_dir") or ""))
    if str(options.get("save_to")) == "original" or str(directory) == ".":
        directory = input_path.parent

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    candidate = directory / f"{stem}{suffix}"
    if overwrite or not candidate.exists():
        return candidate

    for index in range(1, 1000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}_{1000}{suffix}"


# ---------------------------------------------------------------------------
# LUTs
# ---------------------------------------------------------------------------
def available_luts() -> List[Dict[str, str]]:
    """``.cube`` files the user dropped into the LUT folder."""
    found: List[Dict[str, str]] = []
    try:
        for path in sorted(luts_dir().iterdir()):
            if path.is_file() and path.suffix.lower() in _LUT_SUFFIXES:
                found.append({"value": str(path), "label": path.name})
    except OSError:  # pragma: no cover - an unreadable folder is not fatal
        logger.debug("Could not list the LUT folder", exc_info=True)
    return found


def _valid_lut(path_value: str) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_file() and path.suffix.lower() in _LUT_SUFFIXES:
        return path
    return None


# ---------------------------------------------------------------------------
# Form schema for the dialog
# ---------------------------------------------------------------------------
def _field(**kwargs: Any) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def export_form_schema(
    settings: Any,
    *,
    input_name: Optional[str] = None,
    scale: Optional[int] = None,
) -> Dict[str, Any]:
    """Everything the Export dialog needs: fields, labels, defaults, rules.

    ``settings`` is the application settings model; the dialog opens with the
    values the user used last time because they are persisted there.
    """
    from .browse_service import native_dialogs_available

    export = settings.export
    ffmpeg = ffmpeg_path()
    nvidia = has_nvidia_encoder()

    defaults = _defaults(settings, input_name=input_name, scale=scale)

    fields: List[Dict[str, Any]] = [
        _field(
            key="filename",
            label="Filename",
            control="text",
            default=defaults["filename"],
            help="Without the extension - the container or image format decides that.",
        ),
        _field(
            key="save_to",
            label="Save to",
            control="select",
            default=defaults["save_to"],
            options=[
                _field(value=value, label=label, help=help_text)
                for value, label, help_text in SAVE_TARGETS
            ],
        ),
        _field(
            key="export_dir",
            label="Folder",
            control="browse",
            browse="folder",
            default=defaults["export_dir"],
            depends_on={"save_to": ["custom"]},
            help="Where \"Choose a folder\" writes to.",
        ),
        _field(
            key="output_type",
            label="Output type",
            control="segmented",
            default=defaults["output_type"],
            options=[
                _field(value=value, label=label, help=help_text)
                for value, label, help_text in OUTPUT_TYPES
            ],
        ),
        _field(
            key="container",
            label="Container",
            control="select",
            default=defaults["container"],
            options=[
                _field(
                    value=value,
                    label=label,
                    help=help_text,
                    codecs=[codec.encoder for codec in CODECS if value in codec.containers],
                )
                for value, label, help_text in CONTAINERS
            ],
            depends_on={"output_type": ["video"]},
        ),
        _field(
            key="codec",
            label="Codec",
            control="select",
            default=defaults["codec"],
            options=[
                _field(
                    value=codec.encoder,
                    label=codec.label,
                    help=codec.help,
                    containers=list(codec.containers),
                    profiles=list(codec.profiles),
                    disabled=codec.requires_nvidia and not nvidia,
                    reason=(
                        "Needs an NVIDIA GPU with NVENC, which was not detected"
                        if codec.requires_nvidia and not nvidia
                        else None
                    ),
                )
                for codec in CODECS
            ],
            depends_on={"output_type": ["video"]},
            disabled=ffmpeg is None,
            reason="FFmpeg was not found, so the built-in MP4 writer is used" if ffmpeg is None else None,
        ),
        _field(
            key="profile",
            label="Profile",
            control="select",
            default=defaults["profile"],
            options=[_field(value="", label="Automatic")],
            # The real option list is per codec; the dialog narrows it.
            option_sets={
                codec.encoder: [_field(value="", label="Automatic")]
                + [_field(value=profile, label=profile.capitalize()) for profile in codec.profiles]
                for codec in CODECS
                if codec.profiles
            },
            depends_on={"output_type": ["video"]},
            applies_to_codecs=[codec.encoder for codec in CODECS if codec.profiles],
        ),
        _field(
            key="bitrate_mode",
            label="Bitrate",
            control="segmented",
            default=defaults["bitrate_mode"],
            options=[
                _field(value=value, label=label, help=help_text)
                for value, label, help_text in BITRATE_MODES
            ],
            depends_on={"output_type": ["video"]},
        ),
        _field(
            key="quality",
            label="Quality level",
            control="segmented",
            default=defaults["quality"],
            options=[
                _field(value=value, label=label, help=help_text)
                for value, label, help_text in QUALITY_LEVELS
            ],
            depends_on={"output_type": ["video"], "bitrate_mode": ["dynamic"]},
        ),
        _field(
            key="bitrate_mbps",
            label="Target bitrate",
            control="number",
            unit="Mbps",
            min=CONSTANT_BITRATE_MIN_MBPS,
            max=CONSTANT_BITRATE_MAX_MBPS,
            step=0.5,
            default=defaults["bitrate_mbps"],
            depends_on={"output_type": ["video"], "bitrate_mode": ["constant"]},
        ),
        _field(
            key="audio_mode",
            label="Audio mode",
            control="select",
            default=defaults["audio_mode"],
            options=[
                _field(
                    value=mode["value"],
                    label=mode["label"],
                    help=mode.get("help"),
                    containers=[
                        container
                        for container, modes in AUDIO_MODES_BY_CONTAINER.items()
                        if mode["value"] in modes
                    ],
                )
                for mode in AUDIO_MODES
            ],
            depends_on={"output_type": ["video"]},
        ),
        _field(
            key="image_format",
            label="Image format",
            control="select",
            default=defaults["image_format"],
            options=[_field(value=value, label=label) for value, label in IMAGE_FORMATS],
            depends_on={"output_type": ["image_sequence"]},
        ),
        _field(
            key="jpg_quality",
            label="Image quality",
            control="number",
            min=1,
            max=100,
            step=1,
            default=defaults["jpg_quality"],
            depends_on={"output_type": ["image_sequence"], "image_format": ["jpg", "webp"]},
        ),
        _field(
            key="lut_path",
            label="Apply LUT",
            control="browse",
            browse="file",
            filters=[".cube"],
            default=defaults["lut_path"],
            choices=available_luts(),
            choices_dir=str(luts_dir()),
            help="A .cube 3D LUT applied to the encoded video.",
            depends_on={"output_type": ["video"]},
            disabled=ffmpeg is None,
            reason="Applying a LUT needs FFmpeg" if ffmpeg is None else None,
        ),
    ]

    return {
        "fields": fields,
        "defaults": defaults,
        "ffmpeg": {"available": ffmpeg is not None, "path": ffmpeg},
        "nvidia": nvidia,
        "native_dialogs": native_dialogs_available(),
        "naming_pattern": str(getattr(export, "export_naming", "") or "{name}_enhanced"),
        "luts_dir": str(luts_dir()),
        "luts": available_luts(),
    }


def _defaults(settings: Any, *, input_name: Optional[str], scale: Optional[int]) -> Dict[str, Any]:
    """The dialog's opening values: last used, from the persisted settings."""
    export = settings.export
    input_path = Path(input_name) if input_name else None
    pattern = str(getattr(export, "export_naming", "") or "{name}_enhanced")
    filename = ""
    if input_path is not None:
        filename = render_output_name(
            pattern,
            name=input_path.stem,
            scale=scale if scale is not None else getattr(export, "default_upscale", 1),
        )
    return {
        "filename": filename,
        "save_to": str(getattr(export, "save_to", "exports") or "exports"),
        "export_dir": str(getattr(export, "export_dir", "") or ""),
        "output_type": str(getattr(export, "output_type", "video") or "video"),
        "container": str(getattr(export, "container", FALLBACK_CONTAINER) or FALLBACK_CONTAINER),
        "codec": str(getattr(export, "video_codec", "libx264") or "libx264"),
        "profile": str(getattr(export, "video_profile", "") or ""),
        "bitrate_mode": str(getattr(export, "bitrate_mode", "dynamic") or "dynamic"),
        "bitrate_mbps": float(getattr(export, "video_bitrate_mbps", CONSTANT_BITRATE_DEFAULT_MBPS)),
        "quality": str(getattr(export, "video_quality", "high") or "high"),
        "audio_mode": str(getattr(export, "audio_mode", "copy") or "copy"),
        "image_format": str(getattr(export, "image_format", "png") or "png"),
        "jpg_quality": int(getattr(export, "jpg_quality", 95)),
        "lut_path": str(getattr(export, "lut_path", "") or ""),
    }


# ---------------------------------------------------------------------------
# Validation / normalisation
# ---------------------------------------------------------------------------
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _source_audio_codec(path: Optional[Path]) -> Optional[str]:
    """The audio codec of a source file, via ffprobe (``None`` when unknown)."""
    if not path or not Path(path).exists():
        return None
    probe = ffprobe_path()
    if not probe:
        return None
    try:
        result = subprocess.run(
            [
                probe, "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # pragma: no cover - probing is best effort
        logger.debug("ffprobe failed for %s", path, exc_info=True)
        return None
    codec = (result.stdout or "").strip().splitlines()
    return codec[0].strip() if codec and codec[0].strip() else None


def normalise_options(
    raw: Optional[Dict[str, Any]],
    settings: Any,
    *,
    input_path: Optional[Path] = None,
    allow_sequence: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve a dialog state into the values the encoder will actually use.

    Returns ``(options, warnings)``. Cross-field conflicts (a codec the chosen
    container cannot hold, a LUT that is not there, "copy" audio the muxer
    would refuse) are corrected here and explained in ``warnings``.
    """
    warnings: List[str] = []
    raw = dict(raw or {})
    export = settings.export
    ffmpeg = ffmpeg_path()
    nvidia = has_nvidia_encoder()

    output_type = str(raw.get("output_type") or getattr(export, "output_type", "video") or "video")
    if output_type not in {value for value, _, _ in OUTPUT_TYPES}:
        output_type = "video"
        warnings.append("Unknown output type; exporting video instead.")
    if output_type == "image_sequence" and not allow_sequence:
        # Previews are always a video file that the preview pane can play.
        output_type = "video"
        warnings.append("Image sequences are not previewed; rendering a preview video instead.")

    options: Dict[str, Any] = {
        "output_type": output_type,
        "filename": str(raw.get("filename") or "").strip(),
        "save_to": str(raw.get("save_to") or getattr(export, "save_to", "exports") or "exports"),
        "export_dir": str(raw.get("export_dir") or getattr(export, "export_dir", "") or ""),
        "image_format": str(raw.get("image_format") or getattr(export, "image_format", "png") or "png"),
        "jpg_quality": int(_clamp(float(raw.get("jpg_quality", getattr(export, "jpg_quality", 95)) or 95), 1, 100)),
    }

    if options["save_to"] not in {value for value, _, _ in SAVE_TARGETS}:
        options["save_to"] = "exports"
    if options["image_format"] not in {value for value, _ in IMAGE_FORMATS}:
        options["image_format"] = "png"

    # --- LUT -----------------------------------------------------------
    lut_raw = str(raw.get("lut_path") or "").strip()
    lut = _valid_lut(lut_raw)
    if lut_raw and lut is None:
        warnings.append(f"LUT not found at {lut_raw}; exporting without it.")
    if lut is not None and output_type != "video":
        warnings.append("A LUT only applies to video output; exporting frames without it.")
        lut = None

    if output_type == "image_sequence":
        # Frame sequences are written by the built-in writer (no muxer).
        options.update({
            "lut_path": "",
            "container": options["image_format"],
            "codec": FALLBACK_CODEC,
            "profile": "",
            "quality": "",
            "default_quality": "",
            "bitrate_mode": "",
            "bitrate_mbps": 0.0,
            "audio_mode": "none",
            "crf": 0,
        })
        return options, warnings

    # --- container / codec ---------------------------------------------
    container = str(raw.get("container") or getattr(export, "container", FALLBACK_CONTAINER) or FALLBACK_CONTAINER)
    if container not in CONTAINER_LABELS:
        container = FALLBACK_CONTAINER if ffmpeg is None else "mp4"
        warnings.append("Unknown container; using MP4.")

    if ffmpeg is None:
        if container != FALLBACK_CONTAINER:
            warnings.append(
                "FFmpeg was not found, so the built-in writer is used: output is MP4 (MPEG-4 part 2) without audio."
            )
        container = FALLBACK_CONTAINER

    codec_name = str(raw.get("codec") or getattr(export, "video_codec", "libx264") or "libx264")
    codec = CODECS_BY_ENCODER.get(codec_name)
    if codec is not None and codec.requires_nvidia and not nvidia:
        warnings.append(f"{codec.label} needs an NVIDIA GPU; using H.264 (x264) instead.")
        codec = None

    if codec is None:
        codec = CODECS_BY_ENCODER["libx264"]

    if ffmpeg is None:
        codec = None
    elif container not in codec.containers:
        alternatives = [candidate for candidate in CODECS if container in candidate.containers and (not candidate.requires_nvidia or nvidia)]
        if alternatives:
            warnings.append(
                f"{codec.label} cannot be stored in {CONTAINER_LABELS.get(container, container)}; using {alternatives[0].label}."
            )
            codec = alternatives[0]
        else:
            fallback_container = "mp4" if "mp4" in codec.containers else codec.containers[0]
            warnings.append(
                f"{CONTAINER_LABELS.get(container, container)} cannot hold {codec.label}; using {CONTAINER_LABELS.get(fallback_container, fallback_container)}."
            )
            container = fallback_container

    # --- profile / quality / bitrate -----------------------------------
    if codec is None:
        profile = ""
        quality = ""
        bitrate_mode = "dynamic"
        bitrate_mbps = 0.0
        crf = 0
        if raw.get("quality") and str(raw.get("quality")) not in ("", "None"):
            warnings.append("The built-in writer ignores the codec settings.")
    else:
        profile = str(raw.get("profile") or getattr(export, "video_profile", "") or "")
        if profile and profile not in codec.profiles:
            warnings.append(f"{codec.label} has no '{profile}' profile; using the encoder default.")
            profile = ""

        quality = str(raw.get("quality") or getattr(export, "video_quality", "high") or "high")
        if quality not in {value for value, _, _ in QUALITY_LEVELS}:
            quality = "high"
        crf = dict(codec.quality).get(quality, 23)

        bitrate_mode = str(raw.get("bitrate_mode") or getattr(export, "bitrate_mode", "dynamic") or "dynamic")
        if bitrate_mode not in {value for value, _, _ in BITRATE_MODES}:
            bitrate_mode = "dynamic"
        try:
            bitrate_mbps = float(raw.get("bitrate_mbps", getattr(export, "video_bitrate_mbps", CONSTANT_BITRATE_DEFAULT_MBPS)))
        except (TypeError, ValueError):
            bitrate_mbps = CONSTANT_BITRATE_DEFAULT_MBPS
        bitrate_mbps = round(_clamp(bitrate_mbps, CONSTANT_BITRATE_MIN_MBPS, CONSTANT_BITRATE_MAX_MBPS), 2)

    # --- audio ----------------------------------------------------------
    allowed_audio = AUDIO_MODES_BY_CONTAINER.get(container, ("none",))
    audio_mode = str(raw.get("audio_mode") or getattr(export, "audio_mode", "copy") or "copy")
    if audio_mode not in allowed_audio:
        replacement = allowed_audio[0]
        warnings.append(
            f"{CONTAINER_LABELS.get(container, container)} cannot carry {audio_mode} audio; using {replacement}."
        )
        audio_mode = replacement

    if audio_mode == "copy" and input_path is not None:
        source_codec = _source_audio_codec(input_path)
        if source_codec and source_codec not in CONTAINER_AUDIO_CODECS.get(container, ()):
            warnings.append(
                f"The source track is {source_codec.upper()}, which {CONTAINER_LABELS.get(container, container)} cannot hold; re-encoding it as AAC."
            )
            audio_mode = "aac"

    mode = AUDIO_MODES_BY_VALUE.get(audio_mode, AUDIO_MODES_BY_VALUE["none"])

    options.update({
        "container": container,
        "codec": codec.encoder if codec else FALLBACK_CODEC,
        "profile": profile,
        "quality": quality,
        "bitrate_mode": bitrate_mode,
        "bitrate_mbps": bitrate_mbps,
        "crf": crf,
        "audio_mode": audio_mode,
        "audio_codec": mode.get("codec"),
        "audio_bitrate": mode.get("bitrate", ""),
        "lut_path": str(lut) if lut else "",
    })
    return options, warnings


def summarise(options: Dict[str, Any]) -> str:
    """A one-line description of an export, shown in the UI and task metadata."""
    if options.get("output_type") == "image_sequence":
        return f"{options.get('image_format', 'png').upper()} image sequence"
    codec = CODECS_BY_ENCODER.get(str(options.get("codec")))
    parts = [
        str(options.get("container", "")).upper(),
        codec.label if codec else str(options.get("codec", "")),
    ]
    if options.get("bitrate_mode") == "constant":
        parts.append(f"{options.get('bitrate_mbps')} Mbps")
    elif options.get("quality"):
        parts.append(f"{str(options['quality']).capitalize()} quality")
    if options.get("audio_mode") and options["audio_mode"] != "none":
        parts.append("with audio" if options["audio_mode"] == "copy" else f"{options['audio_mode']} audio")
    else:
        parts.append("no audio")
    if options.get("lut_path"):
        parts.append(f"LUT {Path(options['lut_path']).name}")
    return " - ".join(parts)


def options_to_json(options: Dict[str, Any]) -> str:
    """Serialise normalised options for the task metadata/UI."""
    try:
        return json.dumps(options, default=str)
    except (TypeError, ValueError):  # pragma: no cover - options are plain values
        return "{}"
