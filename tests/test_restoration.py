"""Face Recovery + restoration catalog tests.

Repointed from the removed ``src.processing.face_recovery`` / ``restoration``
modules to the catalog-driven pipelines: which face models this build can
really run, which catalog entries are listed but deliberately not implemented,
the checkpoint filenames the model manager writes, and the parameter clamping.
Checkpoint tests are skipped when the weights are absent.
"""
import unittest
from pathlib import Path

from src.models.registry import get_registry
from src.processing.pipeline.base import ModelLoadError
from src.processing.pipeline.registry import get_pipeline_registry

RUNNABLE_FACE_MODELS = {
    "CodeFormer": "CodeFormer",
    "GFPGAN": "GFPGAN",
    "RestoreFormer": "RestoreFormer",
}

CHECKPOINT_FILES = {
    "CodeFormer": "codeformer.pth",
    "GFPGAN": "GFPGANv1.4.pth",
    "RestoreFormer": "RestoreFormer.pth",
}


class FaceRecoveryCatalogTests(unittest.TestCase):
    def test_only_implemented_models_are_offered(self):
        registry = get_registry()
        self.assertEqual(
            sorted(registry.get_available_model_names("Face Recovery")),
            sorted(RUNNABLE_FACE_MODELS),
        )
        self.assertEqual(registry.get_default_model("Face Recovery"), "CodeFormer")

    def test_each_runnable_model_maps_to_its_architecture_and_module(self):
        registry = get_registry()
        for name, arch in RUNNABLE_FACE_MODELS.items():
            with self.subTest(model=name):
                entry = registry.get_model(name)
                self.assertEqual(entry.arch, arch)
                self.assertTrue(entry.is_implemented)
                self.assertEqual(registry.runtime_config(name, "Face Recovery")["arch"], arch)

    def test_checkpoint_filenames_match_the_catalog_downloads(self):
        """The model manager saves the download under its URL basename, and the
        loader looks for exactly that file."""
        registry = get_registry()
        for name, filename in CHECKPOINT_FILES.items():
            with self.subTest(model=name):
                entry = registry.get_model(name)
                self.assertEqual(Path(entry.download_url).name, filename)

    def test_parameter_surface_is_clamped(self):
        from src.processing.face.pipeline import FaceRecoveryPipeline

        schema = FaceRecoveryPipeline.config.parameter_schema["properties"]
        self.assertEqual(set(schema), {"fidelity_weight", "face_upscale"})

        pipeline = FaceRecoveryPipeline(model_name="CodeFormer", device="cpu",
                                        fidelity_weight=5.0, face_upscale=False)
        self.assertEqual(pipeline.fidelity_weight, 1.0)
        self.assertFalse(pipeline.face_upscale)
        self.assertEqual(
            FaceRecoveryPipeline(model_name="CodeFormer", device="cpu",
                                 fidelity_weight=-3.0).fidelity_weight,
            0.0,
        )

    def test_feature_is_registered_with_the_face_pipeline(self):
        pipeline_class = get_pipeline_registry().get_pipeline_class("Face Recovery")
        self.assertEqual(pipeline_class.config.feature, "Face Recovery")


class UnimplementedFaceModelTests(unittest.TestCase):
    """Catalog entries whose backend is deliberately absent must fail loudly."""

    def test_restoreformer_plus_plus_reports_the_missing_backend(self):
        from src.processing.face.pipeline import FaceRecoveryPipeline

        registry = get_registry()
        entry = registry.get_model("RestoreFormer++")
        self.assertIsNotNone(entry)
        self.assertFalse(entry.is_implemented)
        self.assertNotIn("RestoreFormer++", registry.get_available_model_names("Face Recovery"))

        pipeline = FaceRecoveryPipeline(model_name="RestoreFormer++", device="cpu")
        with self.assertRaises(ModelLoadError) as context:
            pipeline.load_model("models/RestoreFormer++.pth")
        self.assertIn("not bundled", str(context.exception))

    def test_missing_model_file_is_reported(self):
        from src.processing.face.pipeline import FaceRecoveryPipeline

        pipeline = FaceRecoveryPipeline(model_name="CodeFormer", device="cpu")
        with self.assertRaises(ModelLoadError) as context:
            pipeline.load_model(None)
        self.assertIn("No model file found", str(context.exception))


class RestorationCatalogHonestyTests(unittest.TestCase):
    """The catalog lists diffusion restorers and RealBasicVSR; this build does
    not ship their runtimes, so they must never be advertised."""

    def test_multipurpose_restoration_is_not_advertised(self):
        registry = get_registry()
        self.assertEqual(registry.get_available_model_names("Multipurpose Restoration"), [])
        self.assertIsNone(get_pipeline_registry().get_pipeline_class("Multipurpose Restoration"))

    def test_realbasicvsr_is_listed_but_not_implemented(self):
        entry = get_registry().get_model("RealBasicVSR")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.arch, "RealBasicVSR")
        self.assertEqual(entry.requires_module, "src.processing.archs.realbasicvsr_arch")
        self.assertFalse(entry.is_implemented)

    def test_supir_and_diffusion_entries_have_no_runtime(self):
        registry = get_registry()
        for name in ("SUPIR", "SeeSR", "StableSR", "CCSR"):
            with self.subTest(model=name):
                entry = registry.get_model(name)
                self.assertIsNotNone(entry)
                self.assertIsNone(entry.arch)
                self.assertFalse(entry.is_implemented)


class FaceCheckpointTests(unittest.TestCase):
    """The bundled checkpoints load 1:1 into the vendored architectures."""

    def _load(self, model_name):
        from src.processing.face.pipeline import FaceRecoveryPipeline

        checkpoint = Path("models") / CHECKPOINT_FILES[model_name]
        if not checkpoint.exists():
            self.skipTest(f"{model_name} checkpoint not downloaded")

        pipeline = FaceRecoveryPipeline(model_name=model_name, device="cpu")
        self.assertTrue(pipeline.load_model(str(checkpoint)))
        return pipeline

    def test_codeformer_checkpoint_loads_strictly(self):
        self._load("CodeFormer")

    def test_gfpgan_checkpoint_loads_strictly(self):
        self._load("GFPGAN")

    def test_restoreformer_checkpoint_loads_strictly(self):
        self._load("RestoreFormer")


if __name__ == "__main__":
    unittest.main()
