"""Upscale tiling tests.

The upscale pipeline slices images larger than ``tile_size`` into tiles (with a
``tile_pad`` border), runs the upsampler per tile and copies each upscaled tile
back to its own full-image position. These tests drive that stitcher with a
deterministic nearest-neighbour upsampler, so the tile-composed output can be
compared with the whole-image upscale pixel by pixel.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.processing.pipeline.base import ProcessingContext


class NearestNeighborUpsampler:
    """Deterministic stand-in for ``realesrgan.RealESRGANer``."""

    def enhance(self, image, outscale=None):
        scale = outscale or 1
        enlarged = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
        return enlarged, None


def _context(tmp, **kwargs):
    return ProcessingContext(
        input_path=Path(tmp) / "in.png",
        output_path=Path(tmp) / "out.png",
        **kwargs,
    )


class UpscaleTilingTests(unittest.TestCase):
    def _pipeline(self, scale=2, tile_size=3, tile_pad=1):
        from src.processing.upscale.pipeline import UpscalePipeline

        pipeline = UpscalePipeline(
            model_name="Real-ESRGAN", device="cpu", scale=scale, tile_size=tile_size
        )
        # The registry-provided config normally carries these; pin them so the
        # test is independent of the catalog entry's loader hyperparameters.
        pipeline._model_config = {"tile_pad": tile_pad}
        pipeline._upsampler = NearestNeighborUpsampler()
        return pipeline

    def test_tiled_output_places_every_tile_at_its_full_image_position(self):
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape((5, 7, 3))
        expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

        with tempfile.TemporaryDirectory() as tmp:
            output = self._pipeline(scale=2, tile_size=3)._process_image_tiled(
                image, _context(tmp)
            )

        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)

    def test_tiled_output_is_exact_for_non_multiple_tile_sizes(self):
        """Dimensions that are not multiples of the tile size still stitch
        without gaps or overlaps."""
        rng = np.random.default_rng(11)
        for h, w, tile_size in ((11, 13, 4), (17, 5, 6), (9, 9, 3), (32, 33, 8)):
            with self.subTest(shape=(h, w), tile_size=tile_size):
                image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

                with tempfile.TemporaryDirectory() as tmp:
                    output = self._pipeline(scale=2, tile_size=tile_size)._process_image_tiled(
                        image, _context(tmp)
                    )

                np.testing.assert_array_equal(output, expected)

    def test_every_tile_is_visited_once(self):
        """Three tiles across and two down must report 6 progress updates."""
        image = np.zeros((5, 7, 3), np.uint8)
        seen = []

        with tempfile.TemporaryDirectory() as tmp:
            context = _context(tmp, progress_callback=lambda p, m: seen.append((p, m)))
            self._pipeline(scale=2, tile_size=3)._process_image_tiled(image, context)

        self.assertEqual(len(seen), 6)
        self.assertTrue(all("tile" in message.lower() for _, message in seen))
        # Progress is monotonic and stays inside the tiling band.
        progress = [value for value, _ in seen]
        self.assertEqual(progress, sorted(progress))
        self.assertGreaterEqual(progress[0], 10)
        self.assertLessEqual(progress[-1], 100)

    def test_process_image_tiles_only_when_the_image_exceeds_tile_size(self):
        """``_process_image`` routes large images through the stitcher and small
        ones straight through the upsampler."""
        import cv2

        for shape, expected_tiles in (((10, 10, 3), True), ((4, 4, 3), False)):
            with self.subTest(shape=shape):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    image = np.random.default_rng(3).integers(0, 256, shape, dtype=np.uint8)
                    input_path, output_path = tmp / "in.png", tmp / "out.png"
                    self.assertTrue(cv2.imwrite(str(input_path), image))

                    pipeline = self._pipeline(scale=2, tile_size=8)
                    visited = []
                    original = pipeline._process_image_tiled

                    def spy(img, context):
                        visited.append(img.shape)
                        return original(img, context)

                    pipeline._process_image_tiled = spy
                    context = ProcessingContext(input_path=input_path, output_path=output_path)
                    result = pipeline._process_image(context)

                    self.assertTrue(result.success)
                    self.assertEqual(bool(visited), expected_tiles)
                    written = cv2.imread(str(result.output_path))
                    self.assertEqual(written.shape, (shape[0] * 2, shape[1] * 2, 3))
                    expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
                    np.testing.assert_array_equal(written, expected)


if __name__ == "__main__":
    unittest.main()
