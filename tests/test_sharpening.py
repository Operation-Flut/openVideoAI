"""Sharpen / Deblur tests.

Covers the pieces the current pipeline actually ships: the catalog-driven
architecture selection (NAFNet block layouts from ``models.json``, Restormer
defaults), the shared overlap-add tiler that makes ``tile_size``/``overlap``
real controls, the parameter clamping, the unsharp finishing pass and the
``strength`` blend. Checkpoint tests are skipped when the weights are absent -
the suite never downloads anything.
"""
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:  # torch is a hard runtime dependency, but keep collection importable
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

from src.processing.pipeline.base import EnhancementPipeline, ProcessingContext


class IdentityModel:
    """Returns its input unchanged, so tiling/stitching must be lossless."""

    def __call__(self, tensor):
        return tensor


class ConstantModel:
    """Returns a constant image, so ``strength`` blending is measurable."""

    def __init__(self, value):
        self.value = value

    def __call__(self, tensor):
        return tensor.new_full(tensor.shape, self.value)


def _image(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _context(tmp):
    return ProcessingContext(
        input_path=Path(tmp) / "in.png",
        output_path=Path(tmp) / "out.png",
    )


class SharpenTilingTests(unittest.TestCase):
    def _pipeline(self, **kwargs):
        from src.processing.sharpen.pipeline import SharpenPipeline

        kwargs.setdefault("model_name", "Restormer")
        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("unsharp_amount", 0.0)
        return SharpenPipeline(**kwargs)

    def test_tiled_identity_model_preserves_every_pixel(self):
        pipeline = self._pipeline(tile_size=128, overlap=32)
        pipeline._model = IdentityModel()

        for shape in ((256, 256), (250, 250), (614, 88), (300, 500)):
            with self.subTest(shape=shape):
                image = _image(*shape, seed=1)
                with tempfile.TemporaryDirectory() as tmp:
                    output = pipeline._process_image_tiled(image, _context(tmp))
                np.testing.assert_array_equal(output, image)

    def test_tiled_identity_model_handles_non_multiple_of_16_tiles(self):
        """Dimensions that are not multiples of 16 force the model to pad
        internally (NAFNet's ``check_image_size``); the stitched output must
        still be pixel-exact."""
        pipeline = self._pipeline(tile_size=64, overlap=32)
        pipeline._model = IdentityModel()

        image = _image(151, 173, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            output = pipeline._process_image_tiled(image, _context(tmp))

        np.testing.assert_array_equal(output, image)

    def test_hann_window_also_stitches_exactly(self):
        pipeline = self._pipeline(tile_size=128, overlap=64, window="hann")
        pipeline._model = IdentityModel()

        image = _image(140, 180, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            output = pipeline._process_image_tiled(image, _context(tmp))

        np.testing.assert_array_equal(output, image)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            self._pipeline(window="cosine")

    def test_tiling_is_used_only_above_the_tile_size(self):
        from src.processing.sharpen.pipeline import SharpenPipeline

        import cv2

        for shape, expect_tiling in (((400, 520, 3), True), ((64, 64, 3), False)):
            with self.subTest(shape=shape):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    image = _image(shape[0], shape[1], seed=4)
                    input_path, output_path = tmp / "in.png", tmp / "out.png"
                    self.assertTrue(cv2.imwrite(str(input_path), image))

                    pipeline = SharpenPipeline(
                        model_name="Restormer", device="cpu", tile_size=128, overlap=32,
                        unsharp_amount=0.0,
                    )
                    pipeline._model = IdentityModel()
                    tiled_calls = []
                    original = pipeline._process_image_tiled

                    def spy(img, context):
                        tiled_calls.append(img.shape)
                        return original(img, context)

                    pipeline._process_image_tiled = spy
                    result = pipeline.process(
                        ProcessingContext(input_path=input_path, output_path=output_path)
                    )

                    self.assertTrue(result.success)
                    self.assertEqual(bool(tiled_calls), expect_tiling)
                    written = cv2.imread(str(result.output_path))
                    np.testing.assert_array_equal(written, image)


class SharpenParameterTests(unittest.TestCase):
    """Every parameter declared in the catalog schema must do something."""

    def _pipeline(self, **kwargs):
        from src.processing.sharpen.pipeline import SharpenPipeline

        kwargs.setdefault("model_name", "Restormer")
        kwargs.setdefault("device", "cpu")
        return SharpenPipeline(**kwargs)

    def test_overlap_is_clamped_to_a_usable_range(self):
        self.assertEqual(self._pipeline(tile_size=512, overlap=8).overlap, 32)
        self.assertEqual(self._pipeline(tile_size=256, overlap=99999).overlap, 128)
        self.assertEqual(self._pipeline(tile_size=512, overlap=128).overlap, 128)

    def test_tile_size_has_a_floor(self):
        self.assertGreaterEqual(self._pipeline(tile_size=1).tile_size, 64)

    def test_directml_uses_memory_conservative_tile_size(self):
        with patch.object(EnhancementPipeline, "get_device", return_value="privateuseone"):
            pipeline = self._pipeline(tile_size=1024)
        self.assertEqual(pipeline.tile_size, 256)

    def test_defaults_come_from_the_catalog(self):
        """NAFNet's catalog entry declares tileSize 512 and the GoPro layout."""
        from src.models.registry import get_registry

        pipeline = self._pipeline(model_name="NAFNet-GoPro")
        self.assertEqual(pipeline.tile_size, 512)
        self.assertEqual(pipeline._model_config["enc_blk_nums"], [1, 1, 1, 28])

    def test_effective_params_lets_step_parameters_win(self):
        pipeline = self._pipeline(tile_size=512, overlap=64, unsharp_amount=0.25, strength=0.5)
        self.assertEqual(
            pipeline._effective_params(),
            {"strength": 0.5, "tile_size": 512, "overlap": 64, "unsharp_amount": 0.25},
        )

        pipeline.parameters = {"strength": 1.5, "tile_size": 128}
        effective = pipeline._effective_params()
        self.assertEqual(effective["strength"], 1.5)
        self.assertEqual(effective["tile_size"], 128)

    def test_unsharp_amount_is_applied(self):
        import cv2

        pipeline = self._pipeline(unsharp_amount=0.0, unsharp_radius=1.0)
        image = _image(64, 96, seed=5)
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        expected = cv2.addWeighted(image, 1.7, blurred, -0.7, 0)

        np.testing.assert_array_equal(pipeline._apply_unsharp_mask(image), image)
        np.testing.assert_array_equal(pipeline._apply_unsharp_mask(image, amount=0.7), expected)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_strength_blends_model_output_with_the_input(self):
        pipeline = self._pipeline(strength=1.0)
        pipeline._model = ConstantModel(1.0)
        image = _image(32, 32, seed=6)

        full = pipeline._run_model(image)
        np.testing.assert_array_equal(full, np.full_like(image, 255))

        pipeline.strength = 0.5
        half = pipeline._run_model(image)
        np.testing.assert_allclose(half, (image.astype(np.float32) + 255.0) / 2, atol=1)

        pipeline.strength = 0.0
        np.testing.assert_allclose(pipeline._run_model(image), image, atol=1)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_run_model_accepts_tuple_outputs(self):
        class TupleModel:
            def __init__(self):
                self.inner = IdentityModel()

            def __call__(self, tensor):
                return (self.inner(tensor),)

        pipeline = self._pipeline(strength=1.0)
        pipeline._model = TupleModel()
        image = _image(24, 32, seed=7)
        np.testing.assert_array_equal(pipeline._run_model(image), image)


class SharpenCatalogRegistrationTests(unittest.TestCase):
    """The catalog, the registry and the pipeline registry must agree."""

    def test_sharpen_models_are_implemented_and_registered(self):
        from src.models.registry import get_registry
        from src.processing.pipeline.registry import get_pipeline_registry

        registry = get_registry()
        names = registry.get_available_model_names("Sharpen / Deblur")
        self.assertIn("NAFNet-GoPro", names)
        self.assertIn("NAFNet-REDS", names)
        self.assertIn("Restormer", names)

        for name in names:
            with self.subTest(model=name):
                entry = registry.get_model(name)
                self.assertTrue(entry.is_implemented)
                self.assertIn(entry.arch, ("NAFNet", "Restormer"))

        pipeline_class = get_pipeline_registry().get_pipeline_class("Sharpen / Deblur", names[0])
        self.assertEqual(pipeline_class.config.feature, "Sharpen / Deblur")


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class NAFNetArchTests(unittest.TestCase):
    def test_block_layout_follows_the_catalog_entry(self):
        from src.models.registry import get_registry
        from src.processing.archs.nafnet_arch import from_runtime_config

        registry = get_registry()
        expected = {
            "NAFNet-SIDD": ([2, 2, 4, 8], [2, 2, 2, 2], 12),
            "NAFNet-GoPro": ([1, 1, 1, 28], [1, 1, 1, 1], 1),
            "NAFNet-REDS": ([1, 1, 1, 28], [1, 1, 1, 1], 1),
        }
        for name, (enc, dec, middle) in expected.items():
            with self.subTest(model=name):
                net = from_runtime_config(registry.runtime_config(name, "Denoise"))
                self.assertEqual([len(block) for block in net.encoders], enc)
                self.assertEqual([len(block) for block in net.decoders], dec)
                self.assertEqual(len(net.middle_blks), middle)

    def test_forward_preserves_odd_sizes_by_padding_internally(self):
        from src.processing.archs.nafnet_arch import NAFNet

        net = NAFNet(width=8, middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
        self.assertEqual(net.padder_size, 4)
        net.eval()
        with torch.no_grad():
            out = net(torch.rand(1, 3, 7, 5))
        self.assertEqual(tuple(out.shape), (1, 3, 7, 5))


class RestormerProfileTests(unittest.TestCase):
    def test_arch_defaults_match_the_released_profile(self):
        from src.processing.archs.restormer import Restormer

        signature = inspect.signature(Restormer.__init__)
        self.assertEqual(signature.parameters["dim"].default, 48)
        self.assertEqual(signature.parameters["num_refinement_blocks"].default, 4)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_checkpoint_loads_strictly_and_runs_on_odd_sizes(self):
        from src.processing.sharpen.pipeline import SharpenPipeline

        checkpoint = Path("models/single_image_defocus_deblurring.pth")
        if not checkpoint.exists():
            self.skipTest("Restormer checkpoint not downloaded")

        pipeline = SharpenPipeline(
            model_name="Restormer", device="cpu", tile_size=256, overlap=32, unsharp_amount=0.0
        )
        self.assertTrue(pipeline.load_model(str(checkpoint)))

        output = pipeline._run_model(_image(61, 47, seed=8))
        self.assertEqual(output.shape, (61, 47, 3))
        self.assertEqual(output.dtype, np.uint8)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_nafnet_gopro_pipeline_processes_a_file(self):
        import cv2

        from src.processing.sharpen.pipeline import SharpenPipeline

        checkpoint = Path("models/NAFNet-GoPro-width64.pth")
        if not checkpoint.exists():
            self.skipTest("NAFNet-GoPro checkpoint not downloaded")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path, output_path = tmp / "in.png", tmp / "out.png"
            image = _image(96, 128, seed=9)
            self.assertTrue(cv2.imwrite(str(input_path), image))

            pipeline = SharpenPipeline(
                model_name="NAFNet-GoPro", device="cpu", strength=1.0, unsharp_amount=0.0
            )
            self.assertTrue(pipeline.load_model(str(checkpoint)))
            result = pipeline.process(
                ProcessingContext(input_path=input_path, output_path=output_path)
            )

            self.assertTrue(result.success)
            self.assertEqual(result.metadata["model_arch"], "NAFNet")
            self.assertEqual(result.metadata["tile_size"], 512)
            written = cv2.imread(str(result.output_path))
            self.assertEqual(written.shape, image.shape)
            # The model is a real restorer: the output is not a copy of the input.
            self.assertFalse(np.array_equal(written, image))


if __name__ == "__main__":
    unittest.main()
