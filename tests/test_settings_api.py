"""Settings tests.

Repointed from the removed ``src.api`` module-level handlers to the current
``SettingsService`` and ``/settings`` route functions. Covers nested defaults,
partial nested updates that persist across a reload, clamping against the flat
``max_values`` policy the UI sliders use, dotted get/set, reset, export/import
and the error paths (400 on an invalid value/enum).
"""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.api.routes import settings as settings_routes
from src.core.config import DEFAULT_SETTINGS, MAX_SETTINGS
from src.services.settings_service import SettingsService, experimental_features_enabled


class _ServiceTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.settings_path = self.tmp_dir / "settings.json"
        self.service = SettingsService(settings_file=self.settings_path)

    def tearDown(self):
        self._tmp.cleanup()


class SettingsServiceTests(_ServiceTestCase):
    def test_nested_defaults_are_applied(self):
        settings = self.service.get_settings()
        # Defaults describe the nested shape, so they are really applied.
        self.assertEqual(settings.appearance.theme, DEFAULT_SETTINGS["appearance"]["theme"])
        self.assertEqual(settings.export.output_format.value, DEFAULT_SETTINGS["export"]["output_format"])
        # The thread count default is derived from the machine, not a constant.
        self.assertEqual(settings.system.cpu_threads, DEFAULT_SETTINGS["system"]["cpu_threads"])
        self.assertGreaterEqual(settings.system.cpu_threads, 1)

    def test_partial_updates_preserve_existing_values_and_persist(self):
        self.service.update_settings({
            "system": {"max_ram_usage": 32},
            "appearance": {"theme": "light"},
            "export": {"output_format": "png"},
        })
        self.service.update_settings({"system": {"max_vram_usage": 12}})

        settings = self.service.get_settings()
        self.assertEqual(settings.system.max_ram_usage, 32)
        self.assertEqual(settings.system.max_vram_usage, 12)
        self.assertEqual(settings.appearance.theme, "light")
        self.assertEqual(settings.export.output_format.value, "png")

        # Written to disk, so a fresh service (an app restart) sees the values.
        on_disk = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["appearance"]["theme"], "light")
        self.assertEqual(on_disk["system"]["max_ram_usage"], 32)

        reloaded = SettingsService(settings_file=self.settings_path).get_settings()
        self.assertEqual(reloaded.appearance.theme, "light")
        self.assertEqual(reloaded.system.max_ram_usage, 32)
        self.assertEqual(reloaded.export.output_format.value, "png")

    def test_numeric_values_are_clamped_to_the_ui_maximums(self):
        self.service.update_settings({
            "system": {"cpu_threads": 99999, "tile_size": 99999},
            "export": {"sharpen_overlap": 9999, "sharpen_unsharp_amount": 99},
        })

        settings = self.service.get_settings()
        self.assertEqual(settings.system.cpu_threads, MAX_SETTINGS["cpuThreads"])
        self.assertEqual(settings.system.tile_size, MAX_SETTINGS["tileSize"])
        self.assertEqual(settings.export.sharpen_overlap, MAX_SETTINGS["sharpenOverlap"])
        self.assertEqual(settings.export.sharpen_unsharp_amount, MAX_SETTINGS["sharpenUnsharpAmount"])

    def test_validate_settings_reports_nested_keys(self):
        self.assertEqual(self.service.validate_settings({"appearance": {"theme": "light"}}), {})
        self.assertIn("system.tile_size", self.service.validate_settings({"system": {"tile_size": 99999}}))
        self.assertIn("appearance.theme", self.service.validate_settings({"appearance": {"theme": "blue"}}))

    def test_dotted_get_and_set(self):
        self.assertFalse(self.service.get_setting("advanced.experimental_features"))
        self.service.set_setting("advanced.experimental_features", True)
        self.assertTrue(self.service.get_setting("advanced.experimental_features"))
        self.assertIsNone(self.service.get_setting("not.a.setting"))

    def test_experimental_features_helper_reads_the_setting(self):
        with patch("src.services.settings_service.get_settings_service", return_value=self.service):
            self.assertFalse(experimental_features_enabled())
            self.service.update_settings({"advanced": {"experimental_features": True}})
            self.assertTrue(experimental_features_enabled())

        with patch("src.services.settings_service.get_settings_service", side_effect=RuntimeError("boom")):
            # A broken settings read must not enable experimental content.
            self.assertFalse(experimental_features_enabled())

    def test_reset_restores_defaults(self):
        self.service.update_settings({"appearance": {"theme": "light"}})
        settings = self.service.reset_to_defaults()
        self.assertEqual(settings.appearance.theme, DEFAULT_SETTINGS["appearance"]["theme"])
        self.assertEqual(json.loads(self.settings_path.read_text(encoding="utf-8")), {})

    def test_export_import_round_trip(self):
        target = self.tmp_dir / "exported.json"
        self.service.update_settings({"appearance": {"theme": "light"}, "system": {"max_ram_usage": 44}})
        self.assertTrue(self.service.export_settings(target))

        self.service.reset_to_defaults()
        self.assertEqual(self.service.get_settings().appearance.theme, "dark")

        self.assertTrue(self.service.import_settings(target))
        settings = self.service.get_settings()
        self.assertEqual(settings.appearance.theme, "light")
        self.assertEqual(settings.system.max_ram_usage, 44)

        self.assertFalse(self.service.import_settings(self.tmp_dir / "missing.json"))


class SettingsRouteTests(_ServiceTestCase):
    def test_get_settings_returns_max_values_for_the_ui_sliders(self):
        response = asyncio.run(settings_routes.get_settings(settings_service=self.service))

        self.assertEqual(response.settings.appearance.theme, "dark")
        # The UI reads the flat camelCase maxima (max_values.cpuThreads / .tileSize).
        self.assertEqual(response.max_values["cpuThreads"], MAX_SETTINGS["cpuThreads"])
        self.assertEqual(response.max_values["tileSize"], MAX_SETTINGS["tileSize"])

    def test_patch_then_get_round_trips(self):
        updated = asyncio.run(settings_routes.update_settings(
            {"appearance": {"theme": "light"}}, settings_service=self.service
        ))
        self.assertEqual(updated.appearance.theme, "light")

        response = asyncio.run(settings_routes.get_settings(settings_service=self.service))
        self.assertEqual(response.settings.appearance.theme, "light")

    def test_invalid_enum_is_rejected_with_400(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(settings_routes.update_settings(
                {"appearance": {"theme": "blue"}}, settings_service=self.service
            ))

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("appearance.theme", context.exception.detail["errors"])

    def test_out_of_range_value_is_rejected_with_400(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(settings_routes.update_settings(
                {"system": {"tile_size": 99999}}, settings_service=self.service
            ))
        self.assertEqual(context.exception.status_code, 400)

    def test_single_setting_endpoints(self):
        result = asyncio.run(settings_routes.set_setting(
            "advanced.debug_logging", True, settings_service=self.service
        ))
        self.assertEqual(result, {"key": "advanced.debug_logging", "value": True})

        result = asyncio.run(settings_routes.get_setting(
            "advanced.debug_logging", settings_service=self.service
        ))
        self.assertEqual(result, {"key": "advanced.debug_logging", "value": True})

        with self.assertRaises(HTTPException) as context:
            asyncio.run(settings_routes.get_setting("not.a.setting", settings_service=self.service))
        self.assertEqual(context.exception.status_code, 404)

    def test_reset_endpoint(self):
        asyncio.run(settings_routes.update_settings(
            {"appearance": {"theme": "light"}}, settings_service=self.service
        ))
        model = asyncio.run(settings_routes.reset_settings(settings_service=self.service))
        self.assertEqual(model.appearance.theme, DEFAULT_SETTINGS["appearance"]["theme"])

    def test_export_endpoint_returns_a_json_download(self):
        response = asyncio.run(settings_routes.export_settings_file(settings_service=self.service))
        self.assertEqual(response.media_type, "application/json")
        self.assertEqual(response.filename, "openvideoai_settings.json")
        self.assertTrue(Path(response.path).exists())
        Path(response.path).unlink(missing_ok=True)

    def test_import_endpoint_applies_an_uploaded_file(self):
        payload = json.dumps({"appearance": {"theme": "light"}}).encode("utf-8")
        upload = UploadFile(file=io.BytesIO(payload), filename="settings.json")

        result = asyncio.run(settings_routes.import_settings_file(
            file=upload, settings_service=self.service
        ))
        self.assertIn("imported", result["message"].lower())
        self.assertEqual(self.service.get_settings().appearance.theme, "light")

    def test_import_endpoint_rejects_non_json_uploads(self):
        upload = UploadFile(file=io.BytesIO(b"nope"), filename="settings.txt")
        with self.assertRaises(HTTPException) as context:
            asyncio.run(settings_routes.import_settings_file(
                file=upload, settings_service=self.service
            ))
        self.assertEqual(context.exception.status_code, 400)


class ComputeBackendTests(_ServiceTestCase):
    """The Compute Backend picker only offers backends this machine can run."""

    CPU_ONLY = {
        "has_cuda": False, "has_hip": False, "has_dml": False,
        "has_metal": False, "best_backend": "cpu",
    }
    AMD_DIRECTML = {
        "has_cuda": False, "has_hip": False, "has_dml": True,
        "has_metal": False, "best_backend": "dml",
    }

    def _backends(self, capabilities):
        from src.utils.system_resources import get_available_compute_backends

        with patch(
            "src.utils.system_resources.get_platform_capabilities",
            return_value=dict(capabilities),
        ):
            return {backend["value"]: backend for backend in get_available_compute_backends()}

    def test_unavailable_backends_are_not_selectable(self):
        backends = self._backends(self.CPU_ONLY)

        # Auto and CPU always work; every GPU backend is flagged with a reason.
        self.assertTrue(backends["auto"]["available"])
        self.assertTrue(backends["cpu"]["available"])
        for value in ("cuda", "hip", "dml", "metal"):
            with self.subTest(backend=value):
                self.assertFalse(backends[value]["available"])
                self.assertIsNotNone(backends[value]["reason"])

    def test_only_detected_hardware_is_marked_available(self):
        backends = self._backends(self.AMD_DIRECTML)

        self.assertTrue(backends["dml"]["available"])
        self.assertIsNone(backends["dml"]["reason"])
        self.assertTrue(backends["dml"]["recommended"])
        self.assertFalse(backends["cuda"]["available"])
        self.assertFalse(backends["metal"]["available"])
        # Auto advertises what it would resolve to, computed by the backend.
        self.assertIn("DirectML", backends["auto"]["label"])

    def test_settings_route_reports_available_backends(self):
        response = asyncio.run(settings_routes.get_settings(settings_service=self.service))

        values = {backend.value for backend in response.available_backends}
        self.assertTrue({"auto", "cpu"}.issubset(values))
        for backend in response.available_backends:
            if not backend.available:
                self.assertIsNotNone(backend.reason)

    def _resolved_device(self, backend, available=("dml", "cpu")):
        from src.services.settings_service import configured_compute_device

        self.service.set_setting("gpu.compute_backend", backend)
        with patch(
            "src.services.settings_service.get_available_devices",
            return_value=list(available),
        ), patch(
            "src.services.settings_service.get_settings_service",
            return_value=self.service,
        ):
            return configured_compute_device()

    def test_configured_device_resolves_only_usable_backends(self):
        self.assertEqual(self._resolved_device("auto"), "auto")
        self.assertEqual(self._resolved_device("dml"), "dml")
        self.assertEqual(self._resolved_device("cpu"), "cpu")
        # Not present on this machine -> the pipeline decides, torch never sees it.
        self.assertEqual(self._resolved_device("cuda"), "auto")
        self.assertEqual(self._resolved_device("hip"), "auto")
        self.assertEqual(self._resolved_device("metal"), "auto")

    def test_legacy_amd_backend_maps_onto_hip_or_directml(self):
        self.assertEqual(self._resolved_device("amd", ("dml", "cpu")), "dml")
        self.assertEqual(self._resolved_device("amd", ("hip", "cpu")), "hip")

    def test_unavailable_choice_is_not_rewritten_in_the_settings(self):
        # Falling back at runtime must not silently change the stored value; the
        # UI shows it as disabled instead so the user can see the mismatch.
        self._resolved_device("cuda")
        self.assertEqual(self.service.get_setting("gpu.compute_backend").value, "cuda")


class ComputeBackendWiringTests(unittest.TestCase):
    """The Compute Backend setting reaches the pipelines instead of being decorative."""

    def _device_used(self, parameters, configured):
        from src.api.schemas import PipelineStep
        from src.services.processing_service import ProcessingService

        recorded = {}

        class FakePipeline:
            def load_model(self, path):
                return True

        def fake_instantiate(pipeline_class, model_name=None, device="auto",
                             parameters=None, task_id=None):
            recorded["device"] = device
            return FakePipeline()

        service = ProcessingService()
        steps = [PipelineStep(feature="Upscale", model="Real-ESRGAN", parameters=parameters)]
        with patch("src.services.processing_service.instantiate_pipeline",
                   side_effect=fake_instantiate), \
             patch("src.services.processing_service.configured_compute_device",
                   return_value=configured), \
             patch("src.services.settings_service.experimental_features_enabled",
                   return_value=False), \
             patch.object(service._model_registry, "get_all_model_paths",
                          return_value={"main": self._tmp_pth}):
            service._build_chain_from_steps(steps, "task1")
        return recorded["device"]

    def setUp(self):
        self._tmp_pth = str(Path("models") / "fake.pth")

    def test_step_device_wins_over_the_setting(self):
        self.assertEqual(
            self._device_used({"scale": 4, "device": "cuda"}, configured="cpu"),
            "cuda",
        )

    def test_setting_is_used_when_the_step_does_not_specify_a_device(self):
        self.assertEqual(self._device_used({"scale": 4}, configured="cpu"), "cpu")
        self.assertEqual(self._device_used({"scale": 4}, configured="auto"), "auto")


if __name__ == "__main__":
    unittest.main()
