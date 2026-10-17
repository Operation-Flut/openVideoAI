"""
Match Color tests: the built-in Reinhard transfer (mean/std matching), the
``preserve_luminance``/``strength`` controls, image + video processing, the
fail-fast reference requirement, the catalog wiring of the other architectures,
and a guard that every endpoint the UI calls exists in the backend.
"""
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.processing.pipeline.base import ProcessingContext, UnsupportedFormatError


def _write_image(path: Path, img: np.ndarray) -> None:
    assert cv2.imwrite(str(path), img)


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


class MatchColorPipelineTests(unittest.TestCase):
    def _pipeline(self, **kwargs):
        from src.processing.color.pipeline import MatchColorPipeline

        pipeline = MatchColorPipeline(**kwargs)
        self.assertTrue(pipeline.load_model(None))
        return pipeline

    def _run_image(self, target, reference, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tp, rp, op = tmp / "target.png", tmp / "ref.png", tmp / "out.png"
            _write_image(tp, target)
            _write_image(rp, reference)

            pipeline = self._pipeline(**kwargs)
            result = pipeline.process(
                ProcessingContext(
                    input_path=tp,
                    output_path=op,
                    parameters={"reference_path": str(rp)},
                )
            )

            self.assertTrue(result.success)
            return cv2.imread(str(result.output_path)), result

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
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tp = tmp / "t.png"
            _write_image(tp, self._tinted(128, 140, 128, 120, 0))

            pipeline = self._pipeline()
            with self.assertRaises(UnsupportedFormatError):
                pipeline.process(
                    ProcessingContext(input_path=tp, output_path=tmp / "o.png")
                )

    def test_missing_reference_file_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tp = tmp / "t.png"
            _write_image(tp, self._tinted(128, 140, 128, 120, 0))

            pipeline = self._pipeline()
            with self.assertRaises(UnsupportedFormatError) as context:
                pipeline.process(
                    ProcessingContext(
                        input_path=tp,
                        output_path=tmp / "o.png",
                        parameters={"reference_path": str(tmp / "nope.png")},
                    )
                )
            self.assertIn("not found", str(context.exception))

    def test_strength_is_clamped_to_the_declared_range(self):
        from src.processing.color.pipeline import MatchColorPipeline

        self.assertEqual(MatchColorPipeline(strength=5.0).strength, 1.0)
        self.assertEqual(MatchColorPipeline(strength=-2.0).strength, 0.0)

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

        self.assertEqual(result.metadata["strength"], 0.0)
        # Strength 0 keeps the target; only the LAB round-trip rounding shows.
        np.testing.assert_allclose(out, target, atol=2.0)

    def test_strength_interpolates_between_target_and_full_transfer(self):
        target = self._tinted(120, 150, 120, 110, 7)
        reference = self._tinted(150, 100, 130, 170, 8)

        full, _ = self._run_image(target, reference, strength=1.0)
        half, _ = self._run_image(target, reference, strength=0.5)

        # The half-strength result is bounded per pixel by both endpoints ...
        low = np.minimum(target, full)
        high = np.maximum(target, full)
        self.assertTrue((half >= low).all(), "half-strength result fell below both endpoints")
        self.assertTrue((half <= high).all(), "half-strength result rose above both endpoints")
        # ... and sits strictly in between on average.
        self.assertLess(float(full.mean()), float(half.mean()))
        self.assertLess(float(half.mean()), float(target.mean()))

    def test_metadata_reports_the_controls_that_were_used(self):
        target = self._tinted(120, 150, 120, 110, 11)
        reference = self._tinted(150, 100, 130, 170, 12)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tp, rp, op = tmp / "target.png", tmp / "ref.png", tmp / "out.png"
            _write_image(tp, target)
            _write_image(rp, reference)

            pipeline = self._pipeline(strength=0.75, preserve_luminance=False)
            result = pipeline.process(
                ProcessingContext(
                    input_path=tp,
                    output_path=op,
                    parameters={"reference_path": str(rp)},
                )
            )

            self.assertEqual(result.metadata["strength"], 0.75)
            self.assertFalse(result.metadata["preserve_luminance"])
            # The reference is recorded as the resolved path, while it still exists.
            self.assertEqual(result.metadata["reference_path"], str(rp))
            self.assertTrue(Path(result.metadata["reference_path"]).exists())

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

            output_path = tmp / "out.mp4"
            pipeline = self._pipeline(strength=1.0, preserve_luminance=True)
            result = pipeline.process(
                ProcessingContext(
                    input_path=video_path,
                    output_path=output_path,
                    parameters={"reference_path": str(ref_path)},
                )
            )

            self.assertTrue(result.success)
            self.assertGreaterEqual(result.metadata["frames_processed"], 4)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


class MatchColorCatalogTests(unittest.TestCase):
    """Match Color models come from the catalog and each declares its runtime."""

    def test_available_models_are_implemented_and_arch_backed(self):
        from src.models.registry import get_registry

        registry = get_registry()
        names = registry.get_available_model_names("Match Color")
        self.assertTrue(names)

        for name in names:
            with self.subTest(model=name):
                entry = registry.get_model(name)
                self.assertTrue(entry.is_implemented)
                self.assertIsNotNone(entry.arch)
                self.assertTrue(entry.has_feature("Match Color"))

    def test_default_model_is_reinhard(self):
        from src.models.registry import get_registry

        self.assertEqual(get_registry().get_default_model("Match Color"), "Reinhard Transfer")


class FrontendBackendContractTests(unittest.TestCase):
    """Every endpoint the UI calls must exist in the backend."""

    _CALL = re.compile(
        r"""\.(?:get|post|patch|put|delete)(?:<[^>]*>)?\(\s*([`'"])(.*?)\1""", re.S
    )

    def test_every_api_path_the_ui_calls_exists_in_the_backend(self):
        from src.api.main import create_app

        source = Path("UI/src/services/api.ts").read_text(encoding="utf-8")
        called = set()
        for _, path in self._CALL.findall(source):
            if not path.startswith("/"):
                continue
            path = path.split("?")[0]
            # Template placeholders (${id}) become the backend's {param} shape.
            path = re.sub(r"\$\{[^}]*\}", "{}", path)
            called.add(path.rstrip("/"))

        self.assertTrue(called, "no API paths found in UI/src/services/api.ts")

        # The OpenAPI document lists every mounted API route with its prefix and
        # parameter shapes, independent of the FastAPI version's route wrapping.
        backend = {
            re.sub(r"\{[^}]*\}", "{}", path).rstrip("/")
            for path in create_app().openapi()["paths"]
        }
        for path in sorted(called):
            with self.subTest(path=path):
                self.assertIn(f"/api{path}", backend)

    def test_processing_route_shape_used_by_the_ui(self):
        from src.api.routes.processing import router

        paths = {getattr(route, "path", "") for route in router.routes}
        self.assertIn("/processing/file", paths)
        self.assertIn("/processing/preview/file", paths)
        self.assertIn("/processing/features", paths)


if __name__ == "__main__":
    unittest.main()
