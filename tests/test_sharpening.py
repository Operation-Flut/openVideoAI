import unittest
from unittest.mock import patch

import numpy as np

from src.processing.sharpening import SHARPEN_MODELS, SharpenPipeline


class SharpenPipelineTests(unittest.TestCase):
    def test_tiled_identity_model_preserves_every_pixel(self):
        import torch

        class IdentityModel:
            def __call__(self, tensor):
                return tensor

        pipeline = SharpenPipeline(device="cpu", tile_size=128)
        pipeline._model = IdentityModel()
        image = np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3).astype(np.uint8)

        with patch.dict("sys.modules", {"torch": torch}):
            output = pipeline._process_image_tiled(image, None)

        np.testing.assert_array_equal(output, image)

    def test_tiled_identity_model_handles_non_multiple_of_16_tiles(self):
        """Image dimensions that are not multiples of 16 force the model to
        pad internally (check_image_size) - the stitched output must still be
        pixel-exact."""
        import torch

        class IdentityModel:
            def __call__(self, tensor):
                return tensor

        pipeline = SharpenPipeline(device="cpu", tile_size=128)
        pipeline._model = IdentityModel()
        image = np.arange(250 * 250 * 3, dtype=np.uint32).reshape(250, 250, 3).astype(np.uint8)

        with patch.dict("sys.modules", {"torch": torch}):
            output = pipeline._process_image_tiled(image, None)

        np.testing.assert_array_equal(output, image)

        # A wide-and-short image exercises the bottom edge tiles as well.
        image = np.arange(614 * 88 * 3, dtype=np.uint32).reshape(614, 88, 3).astype(np.uint8)
        with patch.dict("sys.modules", {"torch": torch}):
            output = pipeline._process_image_tiled(image, None)

        np.testing.assert_array_equal(output, image)

    def test_overlap_add_suppresses_tile_edge_artifacts(self):
        """Hann-window overlap-add blending must remove tile border artifacts.

        A model that corrupts the outer ring of every tile it receives would
        leave visible 48-level seams with hard-cut stitching. The weighted
        blend fades each tile's own border to zero and replaces it with its
        neighbours' window interiors, so the artifacts must be (almost)
        completely suppressed in the stitched output."""
        pipeline = SharpenPipeline(device="cpu", tile_size=128)
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
        artifact = 48

        def noisy_model(tile):
            out = tile.astype(np.float32).copy()
            out[:2] += artifact
            out[-2:] += artifact
            out[:, :2] += artifact
            out[:, -2:] += artifact
            return np.clip(out, 0, 255).astype(np.uint8)

        pipeline._run_model = noisy_model
        output = pipeline._process_image_tiled(image, None)

        error = np.abs(output.astype(int) - image.astype(int))
        self.assertLess(error.mean(), 1.0)
        self.assertLess(error.max(), artifact)

    def test_overlap_default_and_clamping(self):
        pipeline = SharpenPipeline(device="cpu", tile_size=256)
        self.assertEqual(pipeline.overlap, 128)

        # Never overlap more than half the tile (keeps stride >= tile / 2)
        pipeline = SharpenPipeline(device="cpu", tile_size=256, overlap=256)
        self.assertEqual(pipeline.overlap, 128)

        pipeline = SharpenPipeline(device="cpu", tile_size=256, overlap=48)
        self.assertEqual(pipeline.overlap, 48)

        pipeline = SharpenPipeline(device="cpu", tile_size=256, overlap=8)
        self.assertEqual(pipeline.overlap, 32)  # floor

    def test_tiled_identity_exact_with_custom_overlap(self):
        import torch

        class IdentityModel:
            def __call__(self, tensor):
                return tensor

        pipeline = SharpenPipeline(device="cpu", tile_size=256, overlap=96)
        pipeline._model = IdentityModel()
        image = np.arange(400 * 400 * 3, dtype=np.uint32).reshape(400, 400, 3).astype(np.uint8)

        with patch.dict("sys.modules", {"torch": torch}):
            output = pipeline._process_image_tiled(image, None)

        np.testing.assert_array_equal(output, image)

    def test_hamming_window_is_default_and_hann_selectable(self):
        # Hamming only decays to ~0.08 at tile borders (never exact 0.0), which
        # keeps the weight normalisation stable at borders/corners.
        pipeline = SharpenPipeline(device="cpu")
        self.assertEqual(pipeline.window, 'hamming')

        pipeline = SharpenPipeline(device="cpu", window='hann')
        self.assertEqual(pipeline.window, 'hann')

        with self.assertRaises(ValueError):
            SharpenPipeline(device="cpu", window='cosine')

    def test_hann_window_tiling_still_pixel_exact(self):
        import torch

        class IdentityModel:
            def __call__(self, tensor):
                return tensor

        pipeline = SharpenPipeline(device="cpu", tile_size=256, overlap=96, window='hann')
        pipeline._model = IdentityModel()
        image = np.arange(400 * 400 * 3, dtype=np.uint32).reshape(400, 400, 3).astype(np.uint8)

        with patch.dict("sys.modules", {"torch": torch}):
            output = pipeline._process_image_tiled(image, None)

        np.testing.assert_array_equal(output, image)

    def test_unsharp_mask_math_matches_expected_formula(self):
        import cv2

        pipeline = SharpenPipeline(device="cpu", unsharp_amount=0.5, unsharp_radius=1.0)
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (120, 140, 3), dtype=np.uint8)

        result = pipeline._apply_unsharp_mask(image, amount=0.7)
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        expected = cv2.addWeighted(image, 1.7, blurred, -0.7, 0)

        np.testing.assert_array_equal(result, expected)

    def test_unsharp_mask_zero_leaves_image_unchanged(self):
        pipeline = SharpenPipeline(device="cpu", unsharp_amount=0.0)
        image = np.arange(60 * 60 * 3, dtype=np.uint32).reshape(60, 60, 3).astype(np.uint8)
        np.testing.assert_array_equal(pipeline._apply_unsharp_mask(image), image)

    def test_large_image_uses_tiles_without_progress_callback(self):
        pipeline = SharpenPipeline(device="cpu", tile_size=64)
        pipeline._model = object()
        image = np.zeros((80, 96, 3), dtype=np.uint8)

        with patch("src.processing.sharpening.cv2.imread", return_value=image), \
             patch("src.processing.sharpening.cv2.imwrite", return_value=True), \
             patch.object(pipeline, "_process_image_tiled", return_value=image) as process_tiled:
            result = pipeline._process_image("input.png", "output.png")

        self.assertTrue(result["success"])
        process_tiled.assert_called_once()
        self.assertIsNone(process_tiled.call_args.args[1])
        self.assertEqual(result["window"], "hamming")
        self.assertEqual(result["overlap"], 64)
        self.assertIn("unsharp_amount", result)

    def test_directml_uses_memory_conservative_tile_size(self):
        with patch("src.processing.sharpening.EnhancementPipeline._resolve_device", return_value="privateuseone"):
            pipeline = SharpenPipeline()

        self.assertEqual(pipeline.tile_size, 256)

    def test_nafnet_profile_matches_gopro_checkpoint_architecture(self):
        config = SHARPEN_MODELS["NAFNet"]

        self.assertEqual(config["middle_blk_num"], 1)
        self.assertEqual(config["enc_blk_nums"], [1, 1, 1, 28])
        self.assertEqual(config["dec_blk_nums"], [1, 1, 1, 1])

    def test_nafnet_sidd_override_runs_against_the_sidd_checkpoint(self):
        """SharpenPipeline accepts an arch_config override (used by the
        Denoise pipeline for NAFNet-SIDD); here it must build a strict-loadable
        model for the downloaded SIDD checkpoint."""
        from pathlib import Path

        from src.processing.denoising import _NAFNET_SIDD_ARCH_CONFIG

        ckpt = Path("models/NAFNet-SIDD-width64.pth")
        if not ckpt.exists():
            self.skipTest("NAFNet-SIDD checkpoint not downloaded")

        pipeline = SharpenPipeline(
            model_name="NAFNet", device="cpu",
            tile_size=128, arch_config=_NAFNET_SIDD_ARCH_CONFIG,
        )
        self.assertTrue(pipeline.load_model(str(ckpt)))

        tile = np.random.default_rng(3).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        output = pipeline._run_model(tile)
        self.assertEqual(output.shape, (128, 128, 3))
        self.assertEqual(output.dtype, np.uint8)

    def test_restormer_profile_in_sharpen_models(self):
        config = SHARPEN_MODELS["Restormer"]

        self.assertEqual(config["arch"], "restormer")
        self.assertEqual(config["model_file"], "single_image_defocus_deblurring.pth")
        self.assertEqual(config["dim"], 48)
        self.assertEqual(config["num_blocks"], [4, 6, 6, 8])
        self.assertEqual(config["num_refinement_blocks"], 4)

    def test_restormer_architecture_forward_small(self):
        """Restormer builds a working transformer decoder graph on CPU."""
        import torch

        from src.processing.archs.restormer import Restormer

        net = Restormer(
            dim=8, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1
        ).eval()
        with torch.no_grad():
            out = net(torch.rand(1, 3, 96, 96))
            odd = net(torch.rand(1, 3, 61, 47))

        self.assertEqual(tuple(out.shape), (1, 3, 96, 96))
        self.assertEqual(tuple(odd.shape), (1, 3, 61, 47))

    def test_restormer_checkpoint_strict_load_and_run(self):
        """The bundled Restormer defocus checkpoint loads 1:1 and runs.

        Skipped when the weights are not present (never auto-downloaded by
        the test suite).
        """
        from pathlib import Path

        import torch

        ckpt = Path("models/single_image_defocus_deblurring.pth")
        if not ckpt.exists():
            self.skipTest("Restormer defocus checkpoint not downloaded")

        from src.processing.archs.restormer import Restormer

        net = Restormer()
        checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        net.load_state_dict(state, strict=True)

        net.eval()
        with torch.no_grad():
            out = net(torch.rand(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, 3, 224, 224))

    def test_restormer_pipeline_loads_and_runs(self):
        """End-to-end: the sharpen pipeline loads Restormer and runs a tile."""
        from pathlib import Path

        import numpy as np

        ckpt = Path("models/single_image_defocus_deblurring.pth")
        if not ckpt.exists():
            self.skipTest("Restormer defocus checkpoint not downloaded")

        pipeline = SharpenPipeline(model_name="Restormer", device="cpu", tile_size=128)
        self.assertTrue(pipeline.load_model(str(ckpt)))
        tile = np.random.default_rng(4).integers(0, 256, (128, 128, 3), dtype=np.uint8)

        output = pipeline._run_model(tile)

        self.assertEqual(output.shape, (128, 128, 3))
        self.assertEqual(output.dtype, np.uint8)


class NAFNetDirectMLTests(unittest.TestCase):
    def test_forward_preserves_non_square_image_size(self):
        try:
            import torch
            import torch_directml
        except ImportError:
            self.skipTest("PyTorch DirectML is not installed")

        from src.processing.sharpening import NAFNet

        device = torch_directml.device()
        model = NAFNet(
            width=4,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        ).eval().to(device)

        for height, width in ((202, 203), (203, 202), (320, 203)):
            with torch.no_grad():
                output = model(torch.zeros(1, 3, height, width, device=device))

            self.assertEqual(tuple(output.shape[-2:]), (height, width))


if __name__ == "__main__":
    unittest.main()
