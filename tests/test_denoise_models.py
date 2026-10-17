"""
Denoise model integration tests: the bundled SCUNet / NAFNet weights load 1:1
into the vendored architectures and run through the catalog-driven pipeline.
Skipped when the weights are not present (never auto-downloaded by the tests).
"""
import unittest
from pathlib import Path

import numpy as np

from src.processing.pipeline.base import ProcessingContext


class SCUNetWindowAttentionTests(unittest.TestCase):
    """DirectML-safety regressions for the vendored SCUNet architecture.

    DirectML implements neither ``masked_fill`` nor ``roll`` (the former raises
    ``CreateOperator`` failures, the latter silently falls back to the CPU), so
    the shifted-window attention uses an additive ``0``/``-inf`` mask and a
    slice/``cat`` circular shift instead. These tests pin both replacements to
    the exact ops they stand in for.
    """

    def test_roll_spatial_matches_torch_roll(self):
        import torch

        from src.processing.archs.scunet_arch import _roll_spatial

        for h, w in ((8, 8), (16, 11), (6, 12)):
            for shift in (0, 1, 4, -4, 7, -7, h, -h):
                x = torch.rand(2, h, w, 5)
                got = _roll_spatial(x, shift)
                ref = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
                self.assertTrue(torch.equal(got, ref), f"h={h} w={w} shift={shift}")

    def test_wmsa_matches_masked_fill_reference(self):
        """The DML-safe WMSA equals the masked_fill_/roll formulation."""
        import torch
        import torch.nn as nn

        from src.processing.archs.scunet_arch import WMSA

        def reference_forward(mod, x):
            if mod.type != 'W':
                x = torch.roll(
                    x,
                    shifts=(-(mod.window_size // 2), -(mod.window_size // 2)),
                    dims=(1, 2),
                )
            b, h, w, c = x.shape
            p = mod.window_size
            x = x.view(b, h // p, p, w // p, p, c).permute(0, 1, 3, 2, 4, 5)
            hw, ww = x.size(1), x.size(2)
            x = x.reshape(b, hw * ww, p * p, c)
            qkv = mod.embedding_layer(x)
            qkv = qkv.reshape(b, hw * ww, p * p, mod.n_heads * 3, mod.head_dim)
            q, k, v = qkv.permute(3, 0, 1, 2, 4).chunk(3, dim=0)
            sim = torch.matmul(q, k.transpose(-2, -1)) * mod.scale
            sim = sim + mod.relative_embedding().unsqueeze(1).unsqueeze(1)
            if mod.type != 'W':
                mask = torch.zeros(hw, ww, p, p, p, p, dtype=torch.bool)
                s = p - (p // 2)
                mask[-1, :, :s, :, s:, :] = True
                mask[-1, :, s:, :, :s, :] = True
                mask[:, -1, :, :s, :, s:] = True
                mask[:, -1, :, s:, :, :s] = True
                sim = sim.masked_fill_(mask.reshape(1, 1, hw * ww, p * p, p * p), float('-inf'))
            probs = nn.functional.softmax(sim, dim=-1)
            out = torch.matmul(probs, v)
            out = out.permute(1, 2, 3, 0, 4).reshape(
                b, hw * ww, p * p, mod.n_heads * mod.head_dim
            )
            out = mod.linear(out)
            out = (
                out.view(b, hw, ww, p, p, c)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b, hw * p, ww * p, c)
            )
            if mod.type != 'W':
                out = torch.roll(
                    out,
                    shifts=(mod.window_size // 2, mod.window_size // 2),
                    dims=(1, 2),
                )
            return out

        torch.manual_seed(0)
        for block_type in ('W', 'SW'):
            block = WMSA(64, 64, 32, 8, block_type).eval()
            x = torch.rand(1, 32, 24, 64)  # non-square -> exercises per-dim shift
            with torch.no_grad():
                got = block(x)
                ref = reference_forward(block, x)
            self.assertTrue(torch.allclose(got, ref, atol=1e-5), block_type)


class SCUNetModelTests(unittest.TestCase):
    def test_scunet_checkpoint_strict_load_and_run(self):
        """scunet_color_real_psnr.pth loads 1:1 into the vendored SCUNet arch."""
        ckpt = Path("models/scunet_color_real_psnr.pth")
        if not ckpt.exists():
            self.skipTest("SCUNet checkpoint not downloaded")

        import torch

        from src.processing.archs.scunet_arch import SCUNet

        net = SCUNet(
            in_nc=3,
            config=[4, 4, 4, 4, 4, 4, 4],
            dim=64,
            drop_path_rate=0.0,
            input_resolution=512,
        )
        checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        net.load_state_dict(state, strict=True)

        net.eval()
        with torch.no_grad():
            out = net(torch.rand(1, 3, 128, 128))
        self.assertEqual(tuple(out.shape), (1, 3, 128, 128))

    def test_scunet_pipeline_loads_and_runs(self):
        """End-to-end: the Denoise pipeline loads SCUNet and processes a file."""
        import tempfile

        import cv2

        ckpt = Path("models/scunet_color_real_psnr.pth")
        if not ckpt.exists():
            self.skipTest("SCUNet checkpoint not downloaded")

        from src.processing.denoise.pipeline import DenoisePipeline

        pipeline = DenoisePipeline(model_name="SCUNet", device="cpu", strength=0.7)
        self.assertTrue(pipeline.load_model(str(ckpt)))

        input_image = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path, output_path = tmp / "noisy.png", tmp / "clean.png"
            self.assertTrue(cv2.imwrite(str(input_path), input_image))

            result = pipeline.process(
                ProcessingContext(input_path=input_path, output_path=output_path)
            )

            self.assertTrue(result.success)
            self.assertEqual(result.model_used, "SCUNet")
            self.assertEqual(result.metadata["model_arch"], "SCUNet")
            self.assertEqual(result.metadata["strength"], 0.7)
            output = cv2.imread(str(result.output_path))
            self.assertEqual(output.shape, input_image.shape)


class NAFNetSIDDModelTests(unittest.TestCase):
    def test_nafnet_sidd_pipeline_loads_and_runs(self):
        """End-to-end: the Denoise pipeline builds the SIDD block layout from the
        catalog entry, loads the released checkpoint and processes an image."""
        import tempfile
        from pathlib import Path as _Path

        import cv2
        import torch

        ckpt = _Path("models/NAFNet-SIDD-width64.pth")
        if not ckpt.exists():
            self.skipTest("NAFNet-SIDD checkpoint not downloaded")

        from src.processing.denoise.pipeline import DenoisePipeline

        pipeline = DenoisePipeline(model_name="NAFNet-SIDD", device="cpu", strength=0.7)
        self.assertTrue(pipeline.load_model(str(ckpt)))

        # The catalog's runtime block selected the SIDD layout, not the GoPro one.
        model = pipeline._model
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(len(model.middle_blks), 12)
        self.assertEqual([len(e) for e in model.encoders], [2, 2, 4, 8])
        self.assertEqual([len(d) for d in model.decoders], [2, 2, 2, 2])

        tile = np.random.default_rng(0).integers(0, 256, (64, 80, 3), dtype=np.uint8)
        output = pipeline._run_model(tile)
        self.assertEqual(output.shape, (64, 80, 3))
        self.assertEqual(output.dtype, np.uint8)

        input_image = np.random.default_rng(1).integers(0, 256, (64, 80, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = _Path(tmp)
            input_path, output_path = tmp / "noisy.png", tmp / "clean.png"
            self.assertTrue(cv2.imwrite(str(input_path), input_image))

            result = pipeline.process(
                ProcessingContext(input_path=input_path, output_path=output_path)
            )

            self.assertTrue(result.success)
            self.assertEqual(result.metadata["model_arch"], "NAFNet")
            self.assertEqual(cv2.imread(str(result.output_path)).shape, input_image.shape)


if __name__ == "__main__":
    unittest.main()
