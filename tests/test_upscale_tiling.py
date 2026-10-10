import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.processing.upscale import UpscalePipeline


class UpscaleTilingTests(unittest.TestCase):
    def test_tiled_output_places_every_tile_at_its_full_image_position(self):
        class NearestNeighborUpsampler:
            def enhance(self, image, outscale):
                enlarged = np.repeat(
                    np.repeat(image, outscale, axis=0),
                    outscale,
                    axis=1,
                )
                return enlarged, None

        pipeline = UpscalePipeline(scale=2, tile_size=3)
        pipeline._model_config = {"tile_pad": 1}
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape((5, 7, 3))
        expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
        progress = []

        with patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(no_grad=contextlib.nullcontext)},
        ), patch.object(pipeline, "_upsampler", NearestNeighborUpsampler()):
            output = pipeline._process_image_tiled(image, progress.append)

        np.testing.assert_array_equal(output, expected)
        self.assertEqual(len(progress), 6)


if __name__ == "__main__":
    unittest.main()
