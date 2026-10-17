"""
Smart Select - catalog-driven model choice for the Enhancement page.

When the "Smart Select" setting is on, enabling a filter picks the model that
suits the file being processed instead of always taking the catalog default.
The rules live here rather than in the UI because they are catalog and hardware
knowledge, exactly like the compatibility flags the Models page already reads
from the backend:

* only models this build can run (``arch`` present, required module importable)
  and this machine can run (``requiresCuda`` / ``requiresMetal`` / ``canRunOnCpu``
  checked against the detected backend),
* media fit: a model marked ``"media": "video"`` is never chosen for a still
  image, and is preferred for a video (see ``models.json``),
* memory fit: the model has to fit the RAM/VRAM budget the user allowed in
  Settings -> System (``max_ram_usage`` / ``max_vram_usage``),
* already-installed models come first, because processing never downloads on
  demand - a recommendation the user cannot run would be a broken control.

Everything the UI shows as text is produced here too (``reason``, ``hint``), so
the wording stays in one place.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models.registry import ModelConfig, ModelRegistry, get_registry
from ..utils.system_resources import check_model_compatibility, get_system_info

logger = logging.getLogger(__name__)

# Peak footprint of a model relative to its download size. Weight files are
# roughly their own size in memory and activations add to that, so these factors
# are deliberately conservative: they exist to stop a machine being asked to
# load something it cannot hold, not to benchmark it.
RAM_FACTOR = 2.5
VRAM_FACTOR = 1.6

DEFAULT_MAX_RAM_PERCENT = 85
DEFAULT_MAX_VRAM_PERCENT = 95

_SIZE_RE = re.compile(r"([\d.]+)\s*(tb|gb|mb|kb)", re.IGNORECASE)
_SIZE_UNIT_MB = {"kb": 1 / 1024, "mb": 1.0, "gb": 1024.0, "tb": 1024.0 * 1024}

MEDIA_NAMES = ("image", "video")


def parse_size_mb(size: Optional[str]) -> Optional[float]:
    """Download size in MB, or ``None`` when the catalog value is not numeric.

    Handles the display strings used in ``models.json`` such as ``"~65MB"``,
    ``"~6.4GB"`` and ``"Built-in (< 1MB)"``.
    """
    match = _SIZE_RE.search(str(size or ""))
    if not match:
        return None
    return float(match.group(1)) * _SIZE_UNIT_MB[match.group(2).lower()]


def normalise_media(media: Optional[str]) -> Optional[str]:
    """``"image"`` / ``"video"``, or ``None`` when the media is not known.

    ``None`` means "nothing loaded yet": no model is filtered out and no media
    preference is applied, which is the honest answer before a file is chosen.
    """
    value = str(media or "").strip().lower()
    return value if value in MEDIA_NAMES else None


def _settings() -> Any:
    """The live settings model, or ``None`` when it cannot be read."""
    try:
        from .settings_service import get_settings_service

        return get_settings_service().get_settings()
    except Exception:  # pragma: no cover - defensive, settings always exist at runtime
        logger.debug("Could not read settings for Smart Select", exc_info=True)
        return None


def _experimental_enabled() -> bool:
    try:
        from .settings_service import experimental_features_enabled

        return experimental_features_enabled()
    except Exception:  # pragma: no cover
        return False


def _percent(settings: Any, name: str, default: int) -> float:
    value = getattr(getattr(settings, "system", None), name, None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _budgets(settings: Any, system: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """RAM and VRAM budgets in MB, as allowed by the settings sliders.

    ``max_ram_usage`` / ``max_vram_usage`` are percentages of the machine's
    totals (see ``DEFAULT_SETTINGS``), so the System Resources sliders directly
    control how ambitious Smart Select is allowed to be.
    """
    ram_percent = _percent(settings, "max_ram_usage", DEFAULT_MAX_RAM_PERCENT)
    vram_percent = _percent(settings, "max_vram_usage", DEFAULT_MAX_VRAM_PERCENT)

    ram_gb = float(system.get("ram_gb") or 0)
    vram_gb = float(system.get("vram_gb") or 0)

    ram_budget = (ram_gb * 1024 * ram_percent / 100) if ram_gb > 0 else None
    vram_budget = (vram_gb * 1024 * vram_percent / 100) if vram_gb > 0 else None
    return ram_budget, vram_budget


def memory_fit(
    config: ModelConfig,
    *,
    ram_budget_mb: Optional[float],
    vram_budget_mb: Optional[float],
) -> Tuple[bool, Optional[str]]:
    """Whether a model fits the configured memory budgets.

    Returns ``(fits, reason)`` where ``reason`` explains a refusal. Models with
    an unparseable catalog size are never excluded - an unknown size is not
    evidence that it does not fit.
    """
    size_mb = parse_size_mb(getattr(config, "size", None))
    if size_mb is None:
        return True, None

    needed_ram = size_mb * RAM_FACTOR
    if ram_budget_mb and needed_ram > ram_budget_mb:
        return False, (
            f"needs about {needed_ram / 1024:.1f} GB of RAM, "
            f"budget is {ram_budget_mb / 1024:.1f} GB"
        )

    if (config.requires_cuda or config.requires_metal) and vram_budget_mb:
        needed_vram = size_mb * VRAM_FACTOR
        if needed_vram > vram_budget_mb:
            return False, (
                f"needs about {needed_vram / 1024:.1f} GB of VRAM, "
                f"budget is {vram_budget_mb / 1024:.1f} GB"
            )

    return True, None


def _entry(
    registry: ModelRegistry,
    name: str,
    index: int,
    feature: str,
    *,
    ram_budget_mb: Optional[float],
    vram_budget_mb: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Evaluate one candidate, or ``None`` when it cannot run on this machine."""
    config = registry.get_model(name)
    if config is None:
        return None

    compatibility = check_model_compatibility(
        requires_cuda=config.requires_cuda,
        requires_metal=config.requires_metal,
        can_run_on_cpu=config.can_run_on_cpu,
    )
    if not compatibility["compatible"]:
        return None

    fits, fit_reason = memory_fit(
        config, ram_budget_mb=ram_budget_mb, vram_budget_mb=vram_budget_mb
    )

    return {
        "name": name,
        "index": index,
        "config": config,
        "downloaded": bool(registry.is_downloaded(name)),
        "fits": fits,
        "fit_reason": fit_reason,
        "media_match": config.media in MEDIA_NAMES,
        "is_default": feature in (config.default_for or []),
        "size_mb": parse_size_mb(config.size),
    }


def _rank_key(entry: Dict[str, Any], media: Optional[str]) -> tuple:
    """Sort key: installed first, then media fit, AI, catalog default, catalog order."""
    config: ModelConfig = entry["config"]
    return (
        0 if entry["downloaded"] else 1,
        0 if (media and config.media == media) else 1,
        0 if config.is_ai else 1,
        0 if entry["is_default"] else 1,
        entry["index"],
    )


def _fitness_key(entry: Dict[str, Any], media: Optional[str]) -> tuple:
    """Same as :func:`_rank_key` without the "is installed" preference."""
    return _rank_key(entry, media)[1:]


def _describe(entry: Dict[str, Any], feature: str, media: Optional[str], fallbacks: List[str]) -> str:
    """A short, human-readable justification for the chosen model."""
    config: ModelConfig = entry["config"]
    bits: List[str] = []

    if entry["downloaded"]:
        bits.append("already installed")
    else:
        bits.append("not installed yet")

    if media and config.media == media:
        bits.append(f"tuned for {media}")
    elif config.media == "both":
        bits.append("works for stills and video")

    if not config.is_ai:
        bits.append("classical algorithm, no model weights")

    if entry["is_default"]:
        bits.append(f"catalog default for {feature}")

    bits.extend(fallbacks)
    return ", ".join(bits)


def recommend_model(
    feature: str,
    available_models: Sequence[str],
    media: Optional[str] = None,
    *,
    registry: Optional[ModelRegistry] = None,
    system: Optional[Dict[str, Any]] = None,
    settings: Any = None,
) -> Optional[Dict[str, Any]]:
    """Pick the best model for one feature.

    ``available_models`` is the same list the UI's picker offers (the pipeline
    registry's list for the feature), so a recommendation can never name a model
    the picker does not know about. Returns ``None`` when nothing can run here.
    """
    registry = registry or get_registry()
    if not registry.get_all_statuses():
        registry.refresh_status()

    media = normalise_media(media)
    system = system if system is not None else get_system_info()
    settings = settings if settings is not None else _settings()
    ram_budget_mb, vram_budget_mb = _budgets(settings, system)

    candidates = [
        entry
        for index, name in enumerate(available_models)
        if (entry := _entry(
            registry, name, index, feature,
            ram_budget_mb=ram_budget_mb, vram_budget_mb=vram_budget_mb,
        )) is not None
    ]
    if not candidates:
        return None

    fallbacks: List[str] = []

    # Media filter, relaxed when the catalog has nothing for this media type
    # (offering the closest model beats offering none, as long as we say so).
    considered = [e for e in candidates if e["config"].supports_media(media)]
    if not considered:
        considered = candidates
        fallbacks.append(f"no catalog model is marked for {media}")

    # Memory filter, likewise relaxed rather than returning nothing.
    fitting = [e for e in considered if e["fits"]]
    if fitting:
        considered = fitting
    else:
        fallbacks.append("no model fits the configured memory budget")

    chosen = min(considered, key=lambda e: _rank_key(e, media))
    ideal = min(considered, key=lambda e: _fitness_key(e, media))

    hint: Optional[str] = None
    if ideal["name"] != chosen["name"] and not ideal["downloaded"]:
        hint = (
            f"{ideal['name']} is a better fit"
            f"{f' for {media}' if media else ''} - not installed yet."
        )

    return {
        "feature": feature,
        "model": chosen["name"],
        "reason": _describe(chosen, feature, media, fallbacks),
        "media": media or "both",
        "ideal_model": ideal["name"],
        "ideal_installed": ideal["downloaded"],
        "needs_download": not chosen["downloaded"],
        "fallback": bool(fallbacks),
        "hint": hint,
        "candidates": [
            e["name"] for e in sorted(considered, key=lambda e: _rank_key(e, media))
        ],
        "excluded": [
            {"model": e["name"], "reason": e["fit_reason"] or "not compatible with this machine"}
            for e in candidates
            if e["name"] not in {c["name"] for c in considered}
        ],
    }


def recommend_models(
    *,
    features: Optional[Sequence[str]] = None,
    media: Optional[str] = None,
    pipeline_registry: Any = None,
    registry: Optional[ModelRegistry] = None,
    system: Optional[Dict[str, Any]] = None,
    settings: Any = None,
    include_experimental: Optional[bool] = None,
) -> Dict[str, Dict[str, Any]]:
    """Recommendations for every feature the Enhancement sidebar offers.

    Mirrors ``GET /processing/features`` (same feature list, same per-feature
    model list, same experimental gating) so the UI can look a recommendation up
    by feature name without guessing.
    """
    registry = registry or get_registry()
    if pipeline_registry is None:
        from ..processing.pipeline.registry import get_pipeline_registry

        pipeline_registry = get_pipeline_registry()
    if include_experimental is None:
        include_experimental = _experimental_enabled()

    # Read the machine and the settings once for the whole batch.
    system = system if system is not None else get_system_info()
    settings = settings if settings is not None else _settings()

    recommendations: Dict[str, Dict[str, Any]] = {}
    wanted = list(features) if features else pipeline_registry.get_all_features()

    for feature in wanted:
        if not include_experimental and registry.is_feature_experimental(feature):
            continue
        info = pipeline_registry.get_pipeline_info(feature)
        if info is None:
            continue
        available = list(info.available_models)
        if not include_experimental:
            available = registry.drop_experimental(available)
        if not available:
            continue

        try:
            recommendation = recommend_model(
                feature, available, media,
                registry=registry, system=system, settings=settings,
            )
        except Exception:  # pragma: no cover - a bad catalog entry must not 500 the page
            logger.exception("Smart Select failed for feature %r", feature)
            continue
        if recommendation is not None:
            recommendations[feature] = recommendation

    return recommendations
