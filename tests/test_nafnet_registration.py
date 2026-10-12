import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.processing.denoising import DENOISE_MODELS, DenoisePipeline, _NAFNET_SIDD_ARCH_CONFIG
from src.processing.sharpening import SHARPEN_MODELS


class NAFNetRegistrationTests(unittest.TestCase):
    def test_catalog_models_are_registered_with_the_matching_backend(self):
        catalog_path = Path(__file__).resolve().parents[1] / "models.json"
        with catalog_path.open(encoding="utf-8") as catalog_file:
            models = {model["name"]: model for model in json.load(catalog_file)["models"]}

        expected = {
            "NAFNet-SIDD": ("Denoise", DENOISE_MODELS),
            "NAFNet-REDS": ("Sharpen / Deblur", SHARPEN_MODELS),
            "NAFNet-GoPro": ("Sharpen / Deblur", SHARPEN_MODELS),
        }
        for name, (feature, backend_models) in expected.items():
            with self.subTest(model=name):
                model = models[name]
                self.assertIn(feature, [item["name"] for item in model["features"]])
                self.assertIn(name, backend_models)
                self.assertEqual(
                    Path(model["downloadUrl"]).name,
                    backend_models[name]["model_file"],
                )

    def test_nafnet_denoising_uses_tiled_sharpen_pipeline(self):
        pipeline = DenoisePipeline(model_name="NAFNet-SIDD", device="cpu", strength=0.7)
        delegated_pipeline = Mock()
        delegated_pipeline.load_model.return_value = True

        with tempfile.NamedTemporaryFile() as model_file, patch(
            "src.processing.sharpening.SharpenPipeline",
            return_value=delegated_pipeline,
        ) as pipeline_type:
            self.assertTrue(pipeline.load_model(model_file.name))

        pipeline_type.assert_called_once_with(
            model_name="NAFNet",
            device="cpu",
            strength=0.7,
            tile_size=512,
            arch_config=_NAFNET_SIDD_ARCH_CONFIG,
        )
        delegated_pipeline.load_model.assert_called_once()

    def test_nafnet_sidd_arch_config_matches_the_sidd_checkpoint_layout(self):
        """The SIDD checkpoint uses encoders [2,2,4,8], 12 middle blocks and
        decoders [2,2,2,2] -- *not* the GoPro ([1,1,1,28]/1/[1,1,1,1]) layout."""
        self.assertEqual(_NAFNET_SIDD_ARCH_CONFIG["width"], 64)
        self.assertEqual(_NAFNET_SIDD_ARCH_CONFIG["enc_blk_nums"], [2, 2, 4, 8])
        self.assertEqual(_NAFNET_SIDD_ARCH_CONFIG["middle_blk_num"], 12)
        self.assertEqual(_NAFNET_SIDD_ARCH_CONFIG["dec_blk_nums"], [2, 2, 2, 2])

    def test_nafnet_denoising_process_delegates_and_reports_selected_model(self):
        pipeline = DenoisePipeline(model_name="NAFNet-SIDD")
        pipeline._nafnet_pipeline = Mock()
        pipeline._nafnet_pipeline.process.return_value = {"success": True, "model": "NAFNet"}

        result = pipeline.process("input.png", "output.png")

        pipeline._nafnet_pipeline.process.assert_called_once_with("input.png", "output.png")
        self.assertEqual(result["model"], "NAFNet-SIDD")


if __name__ == "__main__":
    unittest.main()
