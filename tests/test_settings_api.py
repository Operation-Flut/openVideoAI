import asyncio
import tempfile
import unittest
from pathlib import Path

from src import api


class SettingsApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_settings_file = api.SETTINGS_FILE
        api.SETTINGS_FILE = Path(self.temp_dir.name) / "settings.json"

    def tearDown(self):
        api.SETTINGS_FILE = self.original_settings_file
        self.temp_dir.cleanup()

    def test_settings_are_persisted_and_partial_updates_preserve_existing_values(self):
        first = asyncio.run(
            api.update_settings(
                api.SettingsModel(maxRamUsage=32, theme="light", outputFormat="png")
            )
        )

        second = asyncio.run(api.update_settings(api.SettingsModel(maxVramUsage=12)))
        settings = asyncio.run(api.get_settings())

        self.assertEqual(first["settings"]["maxRamUsage"], 32)
        self.assertEqual(second["settings"]["maxVramUsage"], 12)
        self.assertEqual(settings["maxRamUsage"], 32)
        self.assertEqual(settings["maxVramUsage"], 12)
        self.assertEqual(settings["theme"], "light")
        self.assertEqual(settings["outputFormat"], "png")

    def test_gpu_acceleration_and_export_settings_are_persisted(self):
        update = asyncio.run(
            api.update_settings(
                api.SettingsModel(
                    computeBackend="cpu",
                    gpuDevice="cpu",
                    exportDir="./custom_exports",
                    exportNaming="{name}_upscaled",
                    videoCodec="libx265",
                    videoCrf=20,
                    overwriteExisting=True
                )
            )
        )
        settings = asyncio.run(api.get_settings())

        self.assertEqual(update["settings"]["computeBackend"], "cpu")
        self.assertEqual(settings["computeBackend"], "cpu")
        self.assertEqual(settings["exportDir"], "./custom_exports")
        self.assertEqual(settings["exportNaming"], "{name}_upscaled")
        self.assertEqual(settings["videoCodec"], "libx265")
        self.assertEqual(settings["videoCrf"], 20)
        self.assertTrue(settings["overwriteExisting"])

    def test_update_export_dir_endpoint(self):
        export_target = str(Path(self.temp_dir.name) / "exports_folder")
        result = asyncio.run(api.update_export_dir({"path": export_target}))
        self.assertEqual(result["path"], export_target)
        settings = asyncio.run(api.get_settings())
        self.assertEqual(settings["exportDir"], export_target)
        self.assertTrue(Path(export_target).exists())

    def test_system_endpoints(self):
        info = asyncio.run(api.get_system_info())
        self.assertIn("cpu", info)
        self.assertIn("ram", info)
        self.assertIn("gpu", info)
        self.assertIn("os", info)

        gpus_response = asyncio.run(api.get_gpus())
        self.assertIn("gpus", gpus_response)
        self.assertIsInstance(gpus_response["gpus"], list)
        self.assertGreater(len(gpus_response["gpus"]), 0)


if __name__ == "__main__":
    unittest.main()

