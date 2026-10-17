"""Smart Select tests.

Covers the catalog-driven ranking behind the "Smart Select" setting
(``src/services/model_recommender.py``) and the endpoint that serves it
(``GET /models/recommendations``).

Hardware detection, the catalog filesystem scan and the settings service are
all stubbed where the rule under test does not need them, so the assertions are
about the rules (media fit, memory budget, installed-first, catalog default)
rather than about the machine running the suite.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.api.routes import models as models_routes
from src.core.config import DEFAULT_SETTINGS
from src.models.registry import ModelConfig, get_registry
from src.services.model_recommender import (
    normalise_media,
    parse_size_mb,
    recommend_model,
    recommend_models,
)
from src.services.settings_service import SettingsService


def make_model(
    name,
    *,
    size="~100MB",
    is_ai=True,
    requires_cuda=False,
    requires_metal=False,
    can_run_on_cpu=True,
    media="both",
    default_for=(),
    features=(),
    experimental=False,
):
    """A catalog entry with only the fields the recommender reads."""
    return ModelConfig(
        name=name,
        description=f"{name} description",
        description_short=f"{name} short",
        size=size,
        weight="Lightweight",
        url="https://example.invalid/model",
        is_ai=is_ai,
        requires_cuda=requires_cuda,
        requires_metal=requires_metal,
        can_run_on_cpu=can_run_on_cpu,
        default_for=list(default_for),
        features=[{"name": feature} for feature in features],
        media=media,
        experimental=experimental,
    )


class FakeRegistry:
    """The slice of the model registry the recommender uses."""

    def __init__(self, models, downloaded=(), experimental_features=()):
        self._models = {model.name: model for model in models}
        self._downloaded = set(downloaded)
        self._experimental_features = set(experimental_features)

    def get_all_statuses(self):
        # Non-empty, so the recommender must not rescan the filesystem.
        return [object()]

    def refresh_status(self):  # pragma: no cover - only runs when statuses are empty
        raise AssertionError("refresh_status should not run when statuses exist")

    def get_model(self, name):
        return self._models.get(name)

    def is_downloaded(self, name):
        return name in self._downloaded

    def is_feature_experimental(self, name):
        return name in self._experimental_features

    def drop_experimental(self, names):
        return [n for n in names if not getattr(self._models.get(n), "experimental", False)]


class FakePipelineRegistry:
    """The slice of the pipeline registry used for the batch call."""

    def __init__(self, features):
        self._features = features

    def get_all_features(self):
        return list(self._features)

    def get_pipeline_info(self, feature):
        if feature not in self._features:
            return None
        return SimpleNamespace(available_models=list(self._features[feature]))


def settings_stub(ram_percent=85, vram_percent=95):
    return SimpleNamespace(
        system=SimpleNamespace(max_ram_usage=ram_percent, max_vram_usage=vram_percent)
    )


def system_stub(ram_gb=16.0, vram_gb=8.0):
    return {"ram_gb": ram_gb, "vram_gb": vram_gb}


def always_compatible(**_kwargs):
    return {"compatible": True, "reason": None}


def cuda_only_is_incompatible(requires_cuda, requires_metal, can_run_on_cpu=True, capabilities=None):
    return {"compatible": not requires_cuda, "reason": "needs CUDA" if requires_cuda else None}


class _RecommenderTestCase(unittest.TestCase):
    def setUp(self):
        # The compatibility flags are covered by the model registry tests; here
        # they are pinned to the machine-independent default.
        patcher = patch("src.services.model_recommender.check_model_compatibility", always_compatible)
        patcher.start()
        self.addCleanup(patcher.stop)

    def recommend(self, feature, models, media, *, registry=None, system=None, settings=None):
        registry = registry or FakeRegistry(models)
        return recommend_model(
            feature,
            [model.name for model in models],
            media,
            registry=registry,
            system=system if system is not None else system_stub(),
            settings=settings if settings is not None else settings_stub(),
        )


class SizeParsingTests(unittest.TestCase):
    def test_catalog_size_strings_are_parsed(self):
        self.assertEqual(parse_size_mb("~65MB"), 65.0)
        self.assertEqual(parse_size_mb("Built-in (< 1MB)"), 1.0)
        self.assertAlmostEqual(parse_size_mb("~6.4GB"), 6.4 * 1024)

    def test_unparseable_sizes_are_unknown_not_zero(self):
        for value in ("Shared system memory", "Unknown", "", None):
            self.assertIsNone(parse_size_mb(value), value)

    def test_media_normalisation(self):
        self.assertEqual(normalise_media("video"), "video")
        self.assertEqual(normalise_media("IMAGE"), "image")
        # "Nothing loaded yet" and "both" both mean: no media preference.
        for value in (None, "", "both", "audio", "video/"):
            self.assertIsNone(normalise_media(value), value)


class MediaRankingTests(_RecommenderTestCase):
    def test_video_model_is_preferred_for_video(self):
        photo = make_model("Photo-Upscaler", default_for=("Upscale",))
        video = make_model("Video-Upscaler", media="video")
        result = self.recommend(
            "Upscale", [photo, video], "video",
            registry=FakeRegistry([photo, video], downloaded={"Photo-Upscaler", "Video-Upscaler"}),
        )
        self.assertEqual(result["model"], "Video-Upscaler")
        self.assertIn("tuned for video", result["reason"])

    def test_video_only_model_is_never_chosen_for_an_image(self):
        photo = make_model("Photo-Upscaler")
        video = make_model("Video-Upscaler", media="video")
        # The video model is listed first: catalog order must not decide this.
        result = self.recommend(
            "Upscale", [video, photo], "image",
            registry=FakeRegistry([photo, video], downloaded={"Photo-Upscaler", "Video-Upscaler"}),
        )
        self.assertEqual(result["model"], "Photo-Upscaler")
        self.assertFalse(result["fallback"])
        self.assertNotIn("Video-Upscaler", result["candidates"])

    def test_image_with_only_video_models_falls_back_and_says_so(self):
        video = make_model("Video-Interpolator", media="video")
        result = self.recommend("Frame Interpolation", [video], "image")
        self.assertEqual(result["model"], "Video-Interpolator")
        self.assertTrue(result["fallback"])
        self.assertIn("no catalog model is marked for image", result["reason"])

    def test_unknown_media_applies_no_preference(self):
        photo = make_model("Photo-Upscaler")
        video = make_model("Video-Upscaler", media="video")
        result = self.recommend(
            "Upscale", [photo, video], None,
            registry=FakeRegistry([photo, video], downloaded={"Photo-Upscaler", "Video-Upscaler"}),
        )
        self.assertEqual(result["media"], "both")
        self.assertFalse(result["fallback"])


class AvailabilityRankingTests(_RecommenderTestCase):
    def test_installed_model_wins_and_the_better_one_is_reported(self):
        photo = make_model("Photo-Upscaler", default_for=("Upscale",))
        video = make_model("Video-Upscaler", media="video")
        result = self.recommend(
            "Upscale", [photo, video], "video",
            registry=FakeRegistry([photo, video], downloaded={"Photo-Upscaler"}),
        )
        self.assertEqual(result["model"], "Photo-Upscaler")
        self.assertFalse(result["needs_download"])
        # The better fit is named rather than silently skipped.
        self.assertEqual(result["ideal_model"], "Video-Upscaler")
        self.assertFalse(result["ideal_installed"])
        self.assertIn("Video-Upscaler", result["hint"])
        self.assertIn("video", result["hint"])

    def test_installed_ideal_model_produces_no_hint(self):
        photo = make_model("Photo-Upscaler")
        video = make_model("Video-Upscaler", media="video")
        result = self.recommend(
            "Upscale", [photo, video], "video",
            registry=FakeRegistry([photo, video], downloaded={"Photo-Upscaler", "Video-Upscaler"}),
        )
        self.assertEqual(result["model"], "Video-Upscaler")
        self.assertIsNone(result["hint"])

    def test_nothing_installed_still_answers_and_flags_the_download(self):
        only = make_model("Only-One")
        result = self.recommend("Upscale", [only], "image")
        self.assertEqual(result["model"], "Only-One")
        self.assertTrue(result["needs_download"])
        self.assertIn("not installed yet", result["reason"])

    def test_catalog_default_breaks_ties_between_equal_candidates(self):
        first = make_model("Alpha")
        marked = make_model("Beta", default_for=("Denoise",))
        result = self.recommend(
            "Denoise", [first, marked], "image",
            registry=FakeRegistry([first, marked], downloaded={"Alpha", "Beta"}),
        )
        self.assertEqual(result["model"], "Beta")
        self.assertIn("catalog default for Denoise", result["reason"])

    def test_ai_model_wins_over_a_classical_algorithm(self):
        classical = make_model("Classical", is_ai=False)
        ai = make_model("AI-One")
        result = self.recommend(
            "Denoise", [classical, ai], "image",
            registry=FakeRegistry([classical, ai], downloaded={"Classical", "AI-One"}),
        )
        self.assertEqual(result["model"], "AI-One")

    def test_model_that_cannot_run_here_is_dropped(self):
        cuda_only = make_model("CUDA-Only", requires_cuda=True, can_run_on_cpu=False)
        cpu_model = make_model("CPU-One")
        registry = FakeRegistry([cuda_only, cpu_model], downloaded={"CUDA-Only", "CPU-One"})
        with patch("src.services.model_recommender.check_model_compatibility", cuda_only_is_incompatible):
            result = self.recommend("Upscale", [cuda_only, cpu_model], "image", registry=registry)
            self.assertEqual(result["model"], "CPU-One")
            self.assertNotIn("CUDA-Only", result["candidates"])
            # Nothing runnable left: no recommendation rather than a broken one.
            self.assertIsNone(self.recommend("Upscale", [cuda_only], "image", registry=registry))


class MemoryBudgetTests(_RecommenderTestCase):
    def test_model_over_the_ram_budget_is_excluded(self):
        huge = make_model("Huge", size="~6.4GB", default_for=("Upscale",))
        small = make_model("Small", size="~65MB")
        result = self.recommend(
            "Upscale", [huge, small], "image",
            registry=FakeRegistry([huge, small], downloaded={"Huge", "Small"}),
            system=system_stub(ram_gb=8.0),
            settings=settings_stub(ram_percent=50),
        )
        # 8 GB * 50% = 4 GB allowed; the 6.4 GB weights need about 16 GB peak.
        self.assertEqual(result["model"], "Small")
        self.assertEqual([entry["model"] for entry in result["excluded"]], ["Huge"])
        self.assertIn("RAM", result["excluded"][0]["reason"])

    def test_gpu_model_over_the_vram_budget_is_excluded(self):
        gpu_only = make_model("Big-GPU", size="~6.4GB", requires_cuda=True, can_run_on_cpu=False)
        cpu_model = make_model("CPU-One", size="~65MB")
        result = self.recommend(
            "Upscale", [gpu_only, cpu_model], "image",
            registry=FakeRegistry([gpu_only, cpu_model], downloaded={"Big-GPU", "CPU-One"}),
            system=system_stub(ram_gb=64.0, vram_gb=8.0),
            settings=settings_stub(vram_percent=50),
        )
        self.assertEqual(result["model"], "CPU-One")
        self.assertEqual([entry["model"] for entry in result["excluded"]], ["Big-GPU"])
        self.assertIn("VRAM", result["excluded"][0]["reason"])

    def test_budget_is_relaxed_rather_than_returning_nothing(self):
        huge = make_model("Huge", size="~6.4GB")
        result = self.recommend(
            "Upscale", [huge], "image",
            registry=FakeRegistry([huge], downloaded={"Huge"}),
            system=system_stub(ram_gb=8.0),
            settings=settings_stub(ram_percent=50),
        )
        self.assertEqual(result["model"], "Huge")
        self.assertTrue(result["fallback"])
        self.assertIn("no model fits the configured memory budget", result["reason"])

    def test_unparseable_size_is_not_treated_as_too_large(self):
        odd = make_model("Odd", size="Shared system memory")
        result = self.recommend("Upscale", [odd], "image")
        self.assertEqual(result["model"], "Odd")
        self.assertFalse(result["fallback"])


class BatchRecommendationTests(_RecommenderTestCase):
    def setUp(self):
        super().setUp()
        self.photo = make_model("Photo-Upscaler", features=("Upscale",))
        self.denoise = make_model("Denoiser", features=("Denoise",))
        self.secret = make_model("Secret", features=("Frame Interpolation",), experimental=True)
        self.registry = FakeRegistry(
            [self.photo, self.denoise, self.secret],
            downloaded={"Photo-Upscaler", "Denoiser", "Secret"},
            experimental_features={"Frame Interpolation"},
        )
        self.pipelines = FakePipelineRegistry({
            "Upscale": ["Photo-Upscaler"],
            "Denoise": ["Denoiser"],
            "Frame Interpolation": ["Secret"],
            "Video Super Resolution": [],
        })

    def batch(self, **kwargs):
        return recommend_models(
            pipeline_registry=self.pipelines,
            registry=self.registry,
            system=system_stub(),
            settings=settings_stub(),
            **kwargs,
        )

    def test_covers_every_feature_that_has_models(self):
        result = self.batch(media="video", include_experimental=False)
        # Features with no available models are not advertised at all.
        self.assertEqual(sorted(result), ["Denoise", "Upscale"])

    def test_single_feature_can_be_requested(self):
        result = self.batch(features=["Denoise"], media="image", include_experimental=False)
        self.assertEqual(list(result), ["Denoise"])

    def test_experimental_feature_is_omitted_unless_enabled(self):
        self.assertNotIn("Frame Interpolation", self.batch(media="video", include_experimental=False))
        self.assertIn("Frame Interpolation", self.batch(media="video", include_experimental=True))

    def test_experimental_model_is_dropped_from_a_picker_when_disabled(self):
        pipelines = FakePipelineRegistry({"Denoise": ["Denoiser", "Secret"]})
        result = recommend_models(
            pipeline_registry=pipelines,
            registry=self.registry,
            media="image",
            system=system_stub(),
            settings=settings_stub(),
            include_experimental=False,
        )
        self.assertEqual(result["Denoise"]["candidates"], ["Denoiser"])


class RecommendationEndpointTests(unittest.TestCase):
    """GET /models/recommendations, called as a coroutine (no HTTP client)."""

    def call(self, **kwargs):
        payload = {
            "Upscale": {
                "feature": "Upscale",
                "model": "Real-ESRGAN",
                "reason": "already installed",
                "media": "video",
                "candidates": ["Real-ESRGAN"],
                "excluded": [],
            }
        }
        with patch("src.services.model_recommender.recommend_models", return_value=payload) as spy, \
                patch("src.services.settings_service.smart_select_enabled", return_value=True), \
                patch("src.services.settings_service.experimental_features_enabled", return_value=False):
            response = asyncio.run(models_routes.get_model_recommendations(**kwargs))
        return response, spy

    def test_payload_carries_the_setting_and_the_recommendations(self):
        response, _ = self.call(media="video", feature="Upscale")
        self.assertTrue(response.smart_select)
        self.assertEqual(response.media, "video")
        self.assertEqual(response.recommendations["Upscale"].model, "Real-ESRGAN")
        self.assertEqual(response.recommendations["Upscale"].reason, "already installed")

    def test_feature_query_narrows_the_batch(self):
        _, spy = self.call(media="video", feature="Upscale")
        self.assertEqual(spy.call_args.kwargs["features"], ["Upscale"])
        _, spy = self.call(media="video", feature=None)
        self.assertIsNone(spy.call_args.kwargs["features"])

    def test_unknown_media_is_reported_as_both(self):
        response, spy = self.call(media="gif", feature=None)
        self.assertEqual(response.media, "both")
        self.assertEqual(spy.call_args.kwargs["media"], "gif")


class CatalogMediaTests(unittest.TestCase):
    """The catalog's `media` field, checked against the real models.json."""

    def test_every_entry_declares_a_known_media(self):
        registry = get_registry()
        models = registry.get_all_models()
        self.assertTrue(models)
        for model in models:
            self.assertIn(model.media, ("image", "video", "both"), model.name)

    def test_video_only_entries_are_marked(self):
        registry = get_registry()
        farneback = [m.name for m in registry.get_all_models() if m.name.startswith("Farneb")]
        self.assertEqual(len(farneback), 1)
        for name in ("RealBasicVSR", "FILM", "RIFE", "SPyNet Optical Flow", "ffmpeg", "handbrake", farneback[0]):
            self.assertEqual(registry.get_media(name), "video", name)

    def test_stills_capable_entries_default_to_both(self):
        registry = get_registry()
        for name in ("Real-ESRGAN", "NAFNet-SIDD", "CodeFormer", "Zero-DCE", "NLUT", "Reinhard Transfer"):
            self.assertEqual(registry.get_media(name), "both", name)

    def test_image_recommendations_never_prefer_a_video_only_model_silently(self):
        registry = get_registry()
        recommendations = recommend_models(media="image", registry=registry, include_experimental=True)
        self.assertTrue(recommendations)
        for feature, recommendation in recommendations.items():
            if registry.get_media(recommendation["model"]) == "video":
                self.assertTrue(
                    recommendation["fallback"],
                    f"{feature} picked a video-only model for an image without saying why",
                )

    def test_video_recommendations_prefer_video_marked_models_when_available(self):
        registry = get_registry()
        recommendations = recommend_models(media="video", registry=registry, include_experimental=True)
        self.assertTrue(recommendations)
        for feature, recommendation in recommendations.items():
            marked = [name for name in recommendation["candidates"] if registry.get_media(name) == "video"]
            if marked and not recommendation["fallback"]:
                self.assertEqual(
                    registry.get_media(recommendation["ideal_model"]),
                    "video",
                    f"{feature} ignored its video-marked candidate(s) {marked}",
                )


class SmartSelectSettingTests(unittest.TestCase):
    """The setting the switch writes, and the defaults/schema agreement."""

    def _service(self, directory):
        return SettingsService(settings_file=Path(directory) / "settings.json")

    def test_defaults_to_off_and_round_trips_across_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            self.assertIn("smart_select", DEFAULT_SETTINGS["models"])
            self.assertFalse(service.get_settings().models.smart_select)

            service.update_settings({"models": {"smart_select": True}})
            self.assertTrue(service.get_settings().models.smart_select)
            # Persisted, so an app restart keeps the user's choice.
            self.assertTrue(self._service(tmp).get_settings().models.smart_select)

    def test_every_default_key_exists_in_the_settings_schema(self):
        # A key present in only one of the two is silently dropped, which is the
        # bug class the nested settings rewrite fixed - so this guards every
        # section, not just the one Smart Select lives in.
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._service(tmp).get_settings()
            for section, values in DEFAULT_SETTINGS.items():
                section_model = getattr(settings, section)
                for key in values:
                    self.assertTrue(
                        hasattr(section_model, key),
                        f"{section}.{key} is missing from the settings schema",
                    )


if __name__ == "__main__":
    unittest.main()
