"""NAFNet catalog/registry/pipeline registration tests.

The catalog is the single source of truth: each NAFNet entry declares the
``NAFNet`` architecture, the optional module its runtime lives in and the block
layout of its released width-64 checkpoint. These tests pin the catalog to the
registry and to the architecture code, so an entry can no longer silently point
at a module that does not exist (the state that used to hide three downloaded
models from every picker).
"""
import unittest
from pathlib import Path

try:  # torch is a hard runtime dependency, but keep collection importable
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

from src.models.registry import ModelConfig, get_registry

NAFNET_MODULE = "src.processing.archs.nafnet_arch"

# Released width-64 block layouts: SIDD uses a wide encoder + middle stack,
# GoPro/REDS use one wide encoder block instead.
EXPECTED_LAYOUTS = {
    "NAFNet-SIDD": ([2, 2, 4, 8], 12, [2, 2, 2, 2]),
    "NAFNet-GoPro": ([1, 1, 1, 28], 1, [1, 1, 1, 1]),
    "NAFNet-REDS": ([1, 1, 1, 28], 1, [1, 1, 1, 1]),
}

EXPECTED_FILES = {
    "NAFNet-SIDD": "NAFNet-SIDD-width64.pth",
    "NAFNet-GoPro": "NAFNet-GoPro-width64.pth",
    "NAFNet-REDS": "NAFNet-REDS-width64.pth",
}


def _entry(name):
    return get_registry().get_model(name)


def _config(**overrides):
    """Build a catalog ``ModelConfig`` with valid required fields."""
    kwargs = dict(
        name="Test-Model",
        description="test",
        description_short="test",
        size="1MB",
        weight="Lightweight",
        url="https://example.invalid/model",
        features=[],
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


class NAFNetCatalogRegistrationTests(unittest.TestCase):
    def test_catalog_models_are_registered_with_the_matching_backend(self):
        for name in EXPECTED_LAYOUTS:
            with self.subTest(model=name):
                entry = _entry(name)
                self.assertIsNotNone(entry, f"{name} missing from the catalog")
                self.assertEqual(entry.arch, "NAFNet")
                self.assertEqual(entry.requires_module, NAFNET_MODULE)
                self.assertTrue(entry.is_implemented)
                self.assertEqual(Path(entry.download_url).name, EXPECTED_FILES[name])

    def test_feature_mapping_matches_the_training_dataset(self):
        self.assertEqual(_entry("NAFNet-SIDD").feature_names, ["Denoise"])
        for name in ("NAFNet-GoPro", "NAFNet-REDS"):
            with self.subTest(model=name):
                self.assertIn("Sharpen / Deblur", _entry(name).feature_names)
                self.assertIn("Denoise", _entry(name).feature_names)

    def test_runtime_block_declares_the_checkpoint_layout(self):
        for name, (enc, middle, dec) in EXPECTED_LAYOUTS.items():
            with self.subTest(model=name):
                runtime = _entry(name).runtime
                self.assertEqual(runtime["encBlkNums"], enc)
                self.assertEqual(runtime["middleBlkNum"], middle)
                self.assertEqual(runtime["decBlkNums"], dec)
                self.assertEqual(runtime["width"], 64)

    def test_registry_resolves_the_arch_and_tunables(self):
        registry = get_registry()
        for name in EXPECTED_LAYOUTS:
            with self.subTest(model=name):
                config = registry.runtime_config(name, "Denoise")
                self.assertEqual(config["arch"], "NAFNet")
                self.assertEqual(config["is_ai"], True)
                self.assertEqual(config["is_classical"], False)
                self.assertEqual(config["enc_blk_nums"], EXPECTED_LAYOUTS[name][0])
                self.assertEqual(config["tile_size"], 512)

    def test_runnable_model_names_include_every_nafnet_entry(self):
        registry = get_registry()
        for feature in ("Denoise", "Sharpen / Deblur"):
            names = registry.get_available_model_names(feature)
            with self.subTest(feature=feature):
                self.assertTrue(names, f"no runnable models for {feature}")
                for name in names:
                    self.assertTrue(registry.get_model(name).is_implemented)
        self.assertIn("NAFNet-SIDD", registry.get_available_model_names("Denoise"))
        self.assertIn("NAFNet-GoPro", registry.get_available_model_names("Sharpen / Deblur"))

    def test_availability_gate_reports_unimplemented_entries(self):
        """The mechanism the pickers rely on: no arch, or a missing module,
        means "listed but not runnable"."""
        catalog_only = _config()
        self.assertIsNone(catalog_only.arch)
        self.assertFalse(catalog_only.is_implemented)

        missing_module = _config(arch="NAFNet", requires_module="src.processing.archs.not_a_real_module")
        self.assertFalse(missing_module.is_implemented)

        present_module = _config(arch="NAFNet", requires_module=NAFNET_MODULE)
        self.assertTrue(present_module.is_implemented)

    def test_pipeline_registry_exposes_both_features(self):
        from src.processing.pipeline.registry import get_pipeline_registry

        registry = get_pipeline_registry()
        for feature in ("Denoise", "Sharpen / Deblur"):
            with self.subTest(feature=feature):
                pipeline_class = registry.get_pipeline_class(feature)
                self.assertEqual(pipeline_class.config.feature, feature)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class NAFNetArchitectureTests(unittest.TestCase):
    def test_arch_module_exposes_the_documented_entry_points(self):
        from src.processing.archs import nafnet_arch

        self.assertTrue(hasattr(nafnet_arch, "NAFNet"))
        self.assertTrue(callable(nafnet_arch.from_runtime_config))

    def test_arch_built_from_the_catalog_uses_the_sidd_layout(self):
        from src.processing.archs.nafnet_arch import from_runtime_config

        net = from_runtime_config(get_registry().runtime_config("NAFNet-SIDD", "Denoise"))
        self.assertEqual([len(block) for block in net.encoders], [2, 2, 4, 8])
        self.assertEqual(len(net.middle_blks), 12)
        self.assertEqual([len(block) for block in net.decoders], [2, 2, 2, 2])

    def test_default_layout_is_gopro_not_sidd(self):
        """A missing runtime block must not silently build a SIDD-shaped net."""
        from src.processing.archs.nafnet_arch import from_runtime_config

        net = from_runtime_config({})
        self.assertEqual([len(block) for block in net.encoders], [1, 1, 1, 28])
        self.assertEqual(len(net.middle_blks), 1)

    def test_sidd_checkpoint_matches_its_own_layout(self):
        checkpoint = Path("models") / EXPECTED_FILES["NAFNet-SIDD"]
        if not checkpoint.exists():
            self.skipTest("NAFNet-SIDD checkpoint not downloaded")

        from src.processing.archs.nafnet_arch import from_runtime_config

        net = from_runtime_config(get_registry().runtime_config("NAFNet-SIDD", "Denoise"))
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = state.get("params") or state.get("params_ema") or state
        # strict=True: a mismatched block layout cannot load.
        net.load_state_dict(state, strict=True)


if __name__ == "__main__":
    unittest.main()
