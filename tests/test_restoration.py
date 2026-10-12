import unittest

from src.processing.face_recovery import FACE_MODELS, get_face_model, list_face_models
from src.processing.restoration import (
    DIFFUSION_RESTORERS,
    MULTIPURPOSE_MODELS,
    MultipurposeRestorationPipeline,
    get_multipurpose_model,
    list_multipurpose_models,
)
from src.processing import ModelLoadError


class FaceRecoveryCatalogTests(unittest.TestCase):
    def test_gfpgan_filename_matches_catalog_download(self):
        """The model manager saves the GFPGAN weight as GFPGANv1.4.pth -
        the pipeline must look for exactly that file."""
        self.assertEqual(FACE_MODELS["GFPGAN"]["model_file"], "GFPGANv1.4.pth")

    def test_face_model_lookup_resolves_default_dir(self):
        """get_face_model resolves its default directory via src.core.paths,
        so it works regardless of the CWD the app was started from."""
        path = get_face_model("CodeFormer")
        # When the repo-relative models/ exists the resolved model dir should
        # point there; either way the lookup must not raise.
        self.assertIsInstance(path, (str, type(None)))
        for entry in list_face_models():
            self.assertIn("name", entry)
            self.assertIn("downloaded", entry)

    def test_codeformer_checkpoint_strict_load(self):
        """The official codeformer.pth weights (params_ema) must strict-load
        into the vendored CodeFormer architecture (guarded: skipped when the
        weights file or torch are not available)."""
        try:
            import torch
            from src.processing.archs.codeformer_arch import CodeFormer
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"CodeFormer arch or torch unavailable: {exc}")
        path = get_face_model("CodeFormer")
        if not path:
            self.skipTest("codeformer.pth not downloaded")
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                         connect_list=['32', '64', '128', '256'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('params_ema') or checkpoint.get('params')
        self.assertIsNotNone(state, "checkpoint has neither params_ema nor params")
        net.load_state_dict(state)  # strict by default

    def test_gfpgan_checkpoint_strict_load(self):
        """The official GFPGANv1.4.pth weights (params_ema, 285 tensors) must
        strict-load into the vendored clean GFPGAN architecture. The arch must
        be the v1.4 'clean' generator (StyleGAN2GeneratorClean base), matching
        the GFPGANer defaults (guarded: skipped when weights are absent)."""
        try:
            import torch
            from src.processing.archs.gfpgan_arch import GFPGANv1Clean
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"GFPGAN arch or torch unavailable: {exc}")
        path = get_face_model("GFPGAN")
        if not path:
            self.skipTest("GFPGANv1.4.pth not downloaded")
        net = GFPGANv1Clean(
            out_size=512, num_style_feat=512, channel_multiplier=2,
            decoder_load_path=None, fix_decoder=False, num_mlp=8,
            input_is_latent=True, different_w=True, narrow=1, sft_half=True)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('params_ema') or checkpoint.get('params')
        self.assertIsNotNone(state, "checkpoint has neither params_ema nor params")
        self.assertEqual(len(state), 285)
        net.load_state_dict(state)  # strict by default

    def test_restoreformer_checkpoint_strict_load(self):
        """The official RestoreFormer.pth weights must strict-load into the
        vendored RestoreFormer architecture (VQGAN encoder + multi-head
        attention transformer decoder). GFPGANer-style fallback: the release
        ships a bare state dict (guarded: skipped when weights are absent)."""
        try:
            import torch
            from src.processing.archs.restoreformer_arch import RestoreFormer
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"RestoreFormer arch or torch unavailable: {exc}")
        path = get_face_model("RestoreFormer")
        if not path:
            self.skipTest("RestoreFormer.pth not downloaded")
        net = RestoreFormer()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('params_ema') or checkpoint.get('params') or checkpoint
        self.assertTrue(state, "checkpoint is empty")
        net.load_state_dict(state)  # strict by default

    def test_restoreformer_plusplus_fails_fast(self):
        """RestoreFormer++ is registered (weights downloadable) but its
        VQGAN+attention backend is not bundled: loading must fail fast with
        a clear message listing the implemented face models."""
        from src.processing.face_recovery import _IMPLEMENTED_FACE_MODELS, FaceRecoveryPipeline
        with self.assertRaises(ModelLoadError) as ctx:
            FaceRecoveryPipeline(model_name="RestoreFormer++").load_model("missing.pth")
        message = str(ctx.exception)
        self.assertIn("RestoreFormer++", message)
        self.assertIn(_IMPLEMENTED_FACE_MODELS, message)


class DiffusionRestorerTests(unittest.TestCase):
    def test_catalog_models_registered_with_downloadable_files(self):
        for name in ("DiffBIR", "SUPIR", "StableSR", "SeeSR", "CCSR", "OSEDiff", "DifFace"):
            config = MULTIPURPOSE_MODELS[name]
            self.assertIn("backend", config)
            self.assertIn(config["backend"], DIFFUSION_RESTORERS)
        # OSEDiff has no direct download URL (Google Drive only), the others do.
        self.assertIsNone(MULTIPURPOSE_MODELS["OSEDiff"]["model_file"])
        for name in ("DiffBIR", "SUPIR", "StableSR", "SeeSR", "CCSR", "DifFace"):
            self.assertIsNotNone(MULTIPURPOSE_MODELS[name]["model_file"])

    def test_diffusion_models_fail_fast_without_backend(self):
        """Selecting a diffusion restorer must raise a clear load error
        instead of silently producing the weak classical estimate."""
        for name, repo_fragment in (("DiffBIR", "XPixelGroup/DiffBIR"), ("DifFace", "czczup/DifFace")):
            with self.assertRaises(ModelLoadError) as ctx:
                pipeline = MultipurposeRestorationPipeline(model_name=name)
                pipeline.load_model()
            message = str(ctx.exception)
            self.assertIn(name, message)
            self.assertIn(repo_fragment, message)

    def test_classical_models_load_trivially(self):
        pipeline = MultipurposeRestorationPipeline(model_name="RealBasicVSR")
        self.assertTrue(pipeline.load_model())
        self.assertEqual(pipeline._backend, "classical")

    def test_multipurpose_model_lookup_and_listing(self):
        self.assertIsNone(get_multipurpose_model("DoesNotExist"))
        # OSEDiff has no model file configured -> cannot be looked up.
        self.assertIsNone(get_multipurpose_model("OSEDiff"))
        entries = list_multipurpose_models()
        names = {entry["name"] for entry in entries}
        self.assertTrue(names >= {"DiffBIR", "SUPIR", "CCSR", "RealBasicVSR"})
        for entry in entries:
            self.assertIn("backend", entry)


if __name__ == "__main__":
    unittest.main()