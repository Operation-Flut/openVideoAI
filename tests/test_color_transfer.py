"""
Match Color tests: the built-in Reinhard transfer (mean/std matching), the
``preserve_luminance``/``strength`` controls, image + video processing, the
fail-fast reference requirement, and a guard that the frontend step -> backend
route mapping stays in sync.
"""
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


def _write_image(path: Path, img: np.ndarray) -> None:
    assert cv2.imwrite(str(path), img)


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


class MatchColorPipelineTests(unittest.TestCase):
    def _run_image(self, target, reference, **kwargs):
        from src.processing.color_transfer import MatchColorPipeline

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tp, rp, op = tmp / "target.png", tmp / "ref.png", tmp / "out.png"
            _write_image(tp, target)
            _write_image(rp, reference)
            pipeline = MatchColorPipeline(**kwargs)
            self.assertTrue(pipeline.load_model())
            result = pipeline.process(str(tp), str(op), reference_path=str(rp))
            self.assertTrue(result["success"])
            return cv2.imread(str(op)), result

    @staticmethod
    def _tinted(base, b, g, r, seed):
        rng = np.random.default_rng(seed)
        img = np.full((72, 96, 3), base, np.uint8)
        img[..., 0] = b
        img[..., 1] = g
        img[..., 2] = r
        noise = rng.integers(-4, 5, img.shape)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def test_requires_a_reference(self):
        from src.processing.color_transfer import MatchColorPipeline
        from src.processing.base import InferenceError

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp) / "t.png"
            _write_image(tp, self._tinted(128, 140, 128, 120, 0))
            pipeline = MatchColorPipeline()
            pipeline.load_model()
            with self.assertRaises(InferenceError):
                pipeline.process(str(tp), str(Path(tmp) / "o.png"), reference_path=None)

    def test_full_transfer_matches_reference_statistics(self):
        target = self._tinted(120, 150, 120, 110, 1)
        reference = self._tinted(150, 100, 130, 170, 2)

        out, _ = self._run_image(target, reference, strength=1.0, preserve_luminance=False)

        # Reinhard matches the per-channel mean exactly (modulo clip/rounding).
        np.testing.assert_allclose(
            _lab(out).mean(axis=(0, 1)),
            _lab(reference).mean(axis=(0, 1)),
            atol=3.0,
        )

    def test_preserve_luminance_keeps_lightness_and_matches_chroma(self):
        target = self._tinted(120, 150, 120, 110, 3)
        reference = self._tinted(150, 100, 130, 170, 4)

        out, _ = self._run_image(target, reference, strength=1.0, preserve_luminance=True)
        out_lab = _lab(out)

        # Lightness stays with the target ...
        np.testing.assert_allclose(
            out_lab[..., 0].mean(), _lab(target)[..., 0].mean(), atol=2.0
        )
        # ... while the a/b chroma channels take the reference grade.
        np.testing.assert_allclose(
            out_lab[..., 1].mean(), _lab(reference)[..., 1].mean(), atol=3.0
        )
        np.testing.assert_allclose(
            out_lab[..., 2].mean(), _lab(reference)[..., 2].mean(), atol=3.0
        )

    def test_strength_zero_returns_the_original(self):
        target = self._tinted(120, 150, 120, 110, 5)
        reference = self._tinted(150, 100, 130, 170, 6)

        out, result = self._run_image(target, reference, strength=0.0)

        self.assertEqual(result["strength"], 0.0)
        np.testing.assert_allclose(out, target, atol=1.0)

    def test_strength_blends_between_input_and_full_transfer(self):
        target = self._tinted(120, 150, 120, 110, 7)
        reference = self._tinted(150, 100, 130, 170, 8)

        full, _ = self._run_image(target, reference, strength=1.0)
        half, _ = self._run_image(target, reference, strength=0.5)

        # The half-strength result sits strictly between input and full grade.
        between = np.clip(
            (target.astype(np.int16) + full.astype(np.int16)) / 2, 0, 255
        )
        np.testing.assert_allclose(half, between, atol=2.0)

    def test_video_is_processed_frame_by_frame(self):
        target_frame = self._tinted(120, 150, 120, 110, 9)
        reference = self._tinted(150, 100, 130, 170, 10)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video_path = tmp / "in.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 72)
            )
            if not writer.isOpened():
                writer.release()
                self.skipTest("OpenCV has no mp4v writer available")

            for _ in range(5):
                writer.write(target_frame)
            writer.release()

            ref_path = tmp / "ref.png"
            _write_image(ref_path, reference)

            from src.processing.color_transfer import MatchColorPipeline

            output_path = tmp / "out.mp4"
            pipeline = MatchColorPipeline(strength=1.0, preserve_luminance=True)
            pipeline.load_model()
            result = pipeline.process(
                str(video_path), str(output_path), reference_path=str(ref_path)
            )

            self.assertTrue(result["success"])
            self.assertGreaterEqual(result["frames_processed"], 4)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


class ProcessingRouteTests(unittest.TestCase):
    """The frontend maps step types to backend route slugs; keep them in sync."""

    def test_frontend_routes_exist_in_backend(self):
        from src import api

        app_js = Path("UI/app.js").read_text(encoding="utf-8")
        match = re.search(r"const PROCESSING_ROUTES = \{(.*?)\};", app_js, re.S)
        self.assertIsNotNone(match, "PROCESSING_ROUTES not found in UI/app.js")

        routes = dict(re.findall(r"(\w+)\s*:\s*'([^']+)'", match.group(1)))
        self.assertEqual(routes.get("color"), "match-color")

        backend_paths = {getattr(r, "path", None) for r in api.app.routes}
        for step_type, slug in routes.items():
            with self.subTest(step=step_type):
                self.assertIn(f"/api/processing/{slug}/file", backend_paths)


if __name__ == "__main__":
    unittest.main()
