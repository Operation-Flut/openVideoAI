"""
Sharpen / Deblur processing.

Primary model: NAFNet (Nonlinear Activation Free Network). The NAFNet
architecture is embedded directly so the pipeline does not depend on
``basicsr.archs.nafnet_arch`` (which is not installed in this project).

The implementation mirrors the official NAFNet-GoPro-width64 checkpoint:
    width         = 64
    enc_blk_nums  = [1, 1, 1, 28]
    middle_blk_num= 1
    dec_blk_nums  = [1, 1, 1, 1]
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard runtime dependency
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from .base import (
    EnhancementPipeline,
    ModelLoadError,
    InferenceError,
    ProcessingAborted,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NAFNet architecture (embedded)
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class LayerNorm2d(nn.Module):
        """Channel-wise LayerNorm for NCHW tensors (as used by NAFNet)."""

        def __init__(self, channels: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.eps = eps

        def forward(self, x):
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / torch.sqrt(var + self.eps)
            return self.weight[None, :, None, None] * y + self.bias[None, :, None, None]

    class SimpleGate(nn.Module):
        """Splits channels in half and multiplies them together."""

        def forward(self, x):
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class NAFBlock(nn.Module):
        """Nonlinear Activation Free block."""

        def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2,
                     drop_out_rate: float = 0.0):
            super().__init__()
            dw_channel = c * DW_Expand
            self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
            self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1,
                                   groups=dw_channel, bias=True)
            self.conv3 = nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True)
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
            )
            self.sg = SimpleGate()
            ffn_channel = FFN_Expand * c
            self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
            self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)

            self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
            self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

            self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        def forward(self, inp):
            x = self.norm1(inp)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.sg(x)
            x = x * self.sca(x)
            x = self.conv3(x)
            x = self.dropout1(x)
            y = inp + x * self.beta

            x = self.norm2(y)
            x = self.conv4(x)
            x = self.sg(x)
            x = self.conv5(x)
            x = self.dropout2(x)
            return y + x * self.gamma

    class NAFNet(nn.Module):
        """NAFNet image restoration network."""

        def __init__(self, img_channel: int = 3, width: int = 16,
                     middle_blk_num: int = 1, enc_blk_nums=None, dec_blk_nums=None):
            super().__init__()
            if enc_blk_nums is None:
                enc_blk_nums = [1, 1, 1, 28]
            if dec_blk_nums is None:
                dec_blk_nums = [1, 1, 1, 1]

            self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
            self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()

            chan = width
            for num in enc_blk_nums:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
                self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
                chan = chan * 2

            self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

            for num in dec_blk_nums:
                self.ups.append(
                    nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
                )
                chan = chan // 2
                self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

            # Pad to a multiple of 2**depth so every downsample/upsample pair
            # lines up exactly (4 levels -> 16).
            self.padder_size = 2 ** len(enc_blk_nums)

        def check_image_size(self, x):
            _, _, h, w = x.size()
            mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
            mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
            if mod_pad_h or mod_pad_w:
                # DirectML's 'reflect' padding applies the height/width pad
                # amounts to the wrong spatial axis for non-square inputs,
                # which breaks the encoder/decoder skip-connection shapes.
                # 'replicate' is implemented correctly there and is a safe,
                # seam-free alternative.
                mode = 'replicate' if x.device.type == 'privateuseone' else 'reflect'
                x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode)
            return x

        def forward(self, inp):
            B, C, H, W = inp.shape
            inp = self.check_image_size(inp)

            x = self.intro(inp)

            encs = []
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encs.append(x)
                x = down(x)

            x = self.middle_blks(x)

            for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
                x = up(x)
                x = x + enc_skip
                x = decoder(x)

            x = self.ending(x)
            x = x + inp

            return x[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Sharpen / Deblur model configurations
# ---------------------------------------------------------------------------
SHARPEN_MODELS: Dict[str, Dict[str, Any]] = {
    'NAFNet': {
        'model_file': 'NAFNet-GoPro-width64.pth',
        'description': 'Blazing-fast image deblurring and sharpness enhancement (NAFNet, GoPro width64).',
        'width': 64,
        'middle_blk_num': 1,
        'enc_blk_nums': [1, 1, 1, 28],
        'dec_blk_nums': [1, 1, 1, 1],
        'tile_size': 512,
    },
    'NAFNet-GoPro': {
        'model_file': 'NAFNet-GoPro-width64.pth',
        'description': 'NAFNet trained on the GoPro dataset for motion deblurring (width64).',
        'width': 64,
        'middle_blk_num': 1,
        'enc_blk_nums': [1, 1, 1, 28],
        'dec_blk_nums': [1, 1, 1, 1],
        'tile_size': 512,
    },
    'NAFNet-REDS': {
        'model_file': 'NAFNet-REDS-width64.pth',
        'description': 'NAFNet trained on the REDS dataset for general video restoration (width64).',
        'width': 64,
        'middle_blk_num': 1,
        'enc_blk_nums': [1, 1, 1, 28],
        'dec_blk_nums': [1, 1, 1, 1],
        'tile_size': 512,
    },
    'NAFNet-URL': {
        'model_file': 'NAFNet-REDS-width64.pth',
        'description': 'NAFNet trained on the REDS dataset for video deblurring (width64).',
        'width': 64,
        'middle_blk_num': 1,
        'enc_blk_nums': [1, 1, 1, 28],
        'dec_blk_nums': [1, 1, 1, 1],
        'tile_size': 512,
    },
    'DeblurGANv2': {
        'model_file': 'deblurganv2.pth',
        'description': 'GAN-based deblurring for severe motion blur.',
        'tile_size': 256,
    },
    'MPRNet': {
        'model_file': 'mprnet_deblurring.pth',
        'description': 'Multi-stage progressive image restoration.',
        'tile_size': 256,
    },
    'Restormer': {
        'model_file': 'single_image_defocus_deblurring.pth',
        'description': 'Transformer-based single-image defocus deblurring (Restormer).',
        'arch': 'restormer',
        'dim': 48,
        'num_blocks': [4, 6, 6, 8],
        'num_refinement_blocks': 4,
        'num_heads': [1, 2, 4, 8],
        'ffn_expansion_factor': 2.66,
        'bias': False,
        'layer_norm_type': 'WithBias',
        'tile_size': 512,
    },
}

# DirectML (AMD/Intel on Windows) has a higher per-dispatch overhead and less
# predictable memory behaviour, so keep its tiles conservative.
_DIRECTML_TILE_CAP = 256


class SharpenPipeline(EnhancementPipeline):
    """Sharpen / deblur enhancement pipeline (NAFNet)."""

    def __init__(self, model_name: str = 'NAFNet', device: str = 'auto',
                 strength: float = 1.0, tile_size: Optional[int] = None,
                 overlap: Optional[int] = None,
                 window: str = 'hamming',
                 unsharp_amount: float = 0.5, unsharp_radius: float = 1.0,
                 arch_config: Optional[Dict[str, Any]] = None):
        super().__init__(device)
        self.model_name = model_name
        self.strength = max(0.0, min(1.0, float(strength)))
        self._model_config = SHARPEN_MODELS.get(model_name, SHARPEN_MODELS['NAFNet'])
        # Optional per-call architecture override (e.g. the Denoise pipeline
        # loads the NAFNet-SIDD weights, whose block layout differs from the
        # GoPro/REDS profile, without registering a duplicate Sharpen model).
        if arch_config:
            merged = dict(self._model_config)
            merged.update(arch_config)
            self._model_config = merged

        if tile_size is None:
            tile_size = int(self._model_config.get('tile_size', 512))
        if str(self.device).startswith('privateuseone'):
            tile_size = min(tile_size, _DIRECTML_TILE_CAP)
        self.tile_size = max(64, int(tile_size))

        # Overlap (px) for the overlap-add stitcher. Capped so the stride stays
        # >= tile_size/2; on DirectML's capped tiles (256) this makes the
        # default 128 the effective maximum.
        self.overlap = max(32, min(
            int(overlap) if overlap is not None else 128,
            max(64, self.tile_size // 2),
        ))

        # Tapering window for the overlap-add stitcher. Hamming is the safe
        # default: it only falls to ~0.08 at the tile borders instead of the
        # exact 0.0 of a Hann window, so the weight normalisation never divides
        # by (near-)zero and border/corner accumulation stays stable.
        if window not in ('hann', 'hamming'):
            raise ValueError(f"window must be 'hann' or 'hamming', got {window!r}")
        self.window = window

        # Optional subtle unsharp-mask finishing after the model output: NAFNet
        # leaves defocus/compression-blurred content rather soft, so a gentle
        # classical sharpening pass restores the perceived sharpness.
        self.unsharp_amount = max(0.0, float(unsharp_amount))
        self.unsharp_radius = max(0.0, float(unsharp_radius))

    # -- model loading ------------------------------------------------------
    def load_model(self, model_path: str, **kwargs) -> bool:
        """Load a sharpening/deblur model from disk."""
        try:
            import torch as _torch

            if not model_path or not Path(model_path).exists():
                raise ModelLoadError(f"Model file not found: {model_path}")

            cfg = self._model_config

            if self.model_name.startswith('NAFNet'):
                model = NAFNet(
                    img_channel=3,
                    width=int(cfg.get('width', 64)),
                    middle_blk_num=int(cfg.get('middle_blk_num', 1)),
                    enc_blk_nums=list(cfg.get('enc_blk_nums', [1, 1, 1, 28])),
                    dec_blk_nums=list(cfg.get('dec_blk_nums', [1, 1, 1, 1])),
                )
                checkpoint = _torch.load(model_path, map_location='cpu', weights_only=False)
                state = checkpoint.get('params', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state, strict=True)
                self._model = model.eval().to(self.device)
            elif cfg.get('arch') == 'restormer':
                from .archs.restormer import Restormer as RestormerNet

                model = RestormerNet(
                    img_channel=3,
                    dim=int(cfg.get('dim', 48)),
                    num_blocks=list(cfg.get('num_blocks', [4, 6, 6, 8])),
                    num_refinement_blocks=int(cfg.get('num_refinement_blocks', 4)),
                    num_heads=list(cfg.get('num_heads', [1, 2, 4, 8])),
                    ffn_expansion_factor=float(cfg.get('ffn_expansion_factor', 2.66)),
                    bias=bool(cfg.get('bias', False)),
                    layer_norm_type=str(cfg.get('layer_norm_type', 'WithBias')),
                )
                checkpoint = _torch.load(model_path, map_location='cpu', weights_only=False)
                state = checkpoint.get('params', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state, strict=True)
                self._model = model.eval().to(self.device)
            else:
                raise ModelLoadError(
                    f"Architecture for '{self.model_name}' is not available in this build"
                )

            logger.info(f"Loaded {self.model_name} deblurring model on {self.device}")
            return True

        except ImportError as e:
            raise ModelLoadError(f"Required dependencies not installed: {e}")
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")

    # -- entry point --------------------------------------------------------
    def process(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process an image or video for sharpening/deblurring."""
        path = Path(input_path)
        if not path.exists():
            raise InferenceError(f"Input file not found: {input_path}")

        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        if path.suffix.lower() in video_extensions:
            return self._process_video(input_path, output_path, **kwargs)
        return self._process_image(input_path, output_path, **kwargs)

    # -- image processing ---------------------------------------------------
    def _process_image(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process a single image for sharpening/deblurring."""
        progress_callback = kwargs.get('progress_callback')

        try:
            if progress_callback:
                progress_callback(10)

            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise InferenceError(f"Failed to read image: {input_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            max_dim = max(h, w)

            if progress_callback:
                progress_callback(30)

            if self._model is not None:
                if max_dim > self.tile_size:
                    output_rgb = self._process_image_tiled(img_rgb, progress_callback)
                else:
                    output_rgb = self._process_image_full(img_rgb)

                # NAFNet preserves size, but guard anyway in case a model pads
                # asymmetrically (keeps cv2.addWeighted from crashing).
                if output_rgb.shape[:2] != img_rgb.shape[:2]:
                    output_rgb = cv2.resize(
                        output_rgb, (img_rgb.shape[1], img_rgb.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                if self.strength < 1.0:
                    output_rgb = cv2.addWeighted(
                        img_rgb, 1.0 - self.strength, output_rgb, self.strength, 0
                    )

                # Optional unsharp-mask finishing, scaled by strength so a low
                # deblur setting also implies a gentle finish.
                if self.unsharp_amount > 0:
                    output_rgb = self._apply_unsharp_mask(
                        output_rgb, self.unsharp_amount * self.strength
                    )
            else:
                output_rgb = self._classical_sharpen(img_rgb)

            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            output_bgr = np.ascontiguousarray(output_bgr)

            if progress_callback:
                progress_callback(80)

            if not cv2.imwrite(output_path, output_bgr):
                raise InferenceError(f"Failed to write output image: {output_path}")

            if progress_callback:
                progress_callback(100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'input_shape': img.shape[:2],
                'output_shape': output_bgr.shape[:2],
                'model': self.model_name,
                'strength': self.strength,
                'overlap': self.overlap,
                'window': self.window,
                'unsharp_amount': self.unsharp_amount,
            }

        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Sharpen/deblur failed: {e}")

    def _run_model(self, tile: np.ndarray) -> np.ndarray:
        """Run the model on an RGB uint8 tile and return RGB uint8 output."""
        import torch as _torch

        tile_tensor = (
            _torch.from_numpy(np.ascontiguousarray(tile))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        with _torch.no_grad():
            out = self._model(tile_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.squeeze(0).permute(1, 2, 0).float().cpu()
        out = out.mul(255.0).clamp(0, 255).round().numpy().astype(np.uint8)
        return out

    def _process_image_full(self, img_rgb: np.ndarray) -> np.ndarray:
        """Process a full image in a single model call."""
        return self._run_model(img_rgb)

    def _process_image_tiled(self, img: np.ndarray, progress_callback: callable) -> np.ndarray:
        """Process a large image in overlapping tiles with windowed blending.

        Classic overlap-add stitching: every tile is fed to the model
        independently, a 2D window (Hamming by default, Hann optionally) fades
        its borders, and the weighted tiles are accumulated and normalised by
        the window sum. Tile seams -- and the model's inference border
        artifacts that used to show up as colourful static blocks at
        edges/corners -- are therefore fully blended away instead of being
        hard-cut into the output.

        Two details keep this pixel-exact and artifact-free:

        * The image is reflected by one ``overlap`` margin *before* tiling
          (numpy-level, DirectML-safe), so border and corner pixels are always
          covered by a neighbouring tile's window interior. Without this they
          would sit on a window zero and render the outer image border black.
        * The default Hamming window only decays to ~0.08 at the tile borders
          (never to the exact 0.0 of a Hann window), so the weight
          normalisation has a guaranteed floor and the border/corner
          accumulation cannot divide by (near-)zero.
        * The final normalisation is rounded to nearest (``rint``) instead of
          truncated, so the float32 weighted average reproduces the input
          exactly for an identity model.
        """
        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        tile_size = max(256, (int(self.tile_size) // 16) * 16)
        overlap = int(self.overlap)
        stride = tile_size - overlap

        # Reflect by ``overlap`` on every side so edge tiles never see a hard
        # image boundary and every original pixel lies inside some window.
        padded_img = cv2.copyMakeBorder(
            img, overlap, overlap, overlap, overlap, cv2.BORDER_REFLECT_101
        )
        padded_h, padded_w = padded_img.shape[:2]

        output = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        def get_window(window_h: int, window_w: int) -> np.ndarray:
            if self.window == 'hann':
                wy = np.hanning(window_h)
                wx = np.hanning(window_w)
            else:
                wy = np.hamming(window_h)
                wx = np.hamming(window_w)
            return np.outer(wy, wx)[..., None].astype(np.float32)

        x_coords = list(range(0, padded_w, stride))
        y_coords = list(range(0, padded_h, stride))
        total_tiles = max(1, len(x_coords) * len(y_coords))
        tile_count = 0

        for yp in y_coords:
            for xp in x_coords:
                x_end = min(xp + tile_size, padded_w)
                y_end = min(yp + tile_size, padded_h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                tile = padded_img[y_start:y_end, x_start:x_end]
                tile_h, tile_w = tile.shape[:2]

                # NAFNet's check_image_size() handles multiple-of-16 padding internally.
                out_tile = self._run_model(tile).astype(np.float32)

                # Hann weight for this tile, fading its borders to zero.
                win = get_window(tile_h, tile_w)

                # Clip accumulation to the original (unpadded) region.
                out_y0 = max(y_start, overlap)
                out_y1 = min(y_end, overlap + h)
                out_x0 = max(x_start, overlap)
                out_x1 = min(x_end, overlap + w)
                if out_y0 >= out_y1 or out_x0 >= out_x1:
                    continue

                output[out_y0 - overlap: out_y1 - overlap,
                       out_x0 - overlap: out_x1 - overlap] += (
                    out_tile[out_y0 - y_start: out_y1 - y_start,
                             out_x0 - x_start: out_x1 - x_start]
                    * win[out_y0 - y_start: out_y1 - y_start,
                          out_x0 - x_start: out_x1 - x_start]
                )
                weight[out_y0 - overlap: out_y1 - overlap,
                       out_x0 - overlap: out_x1 - overlap] += (
                    win[out_y0 - y_start: out_y1 - y_start,
                        out_x0 - x_start: out_x1 - x_start]
                )

                tile_count += 1
                if progress_callback:
                    progress = 30 + int((tile_count / total_tiles) * 50)
                    progress_callback(progress)

        # Normalise by the accumulated window weights.
        weight = np.maximum(weight, 1e-8)
        output = output / weight
        return np.clip(np.rint(output), 0, 255).astype(np.uint8)

    def _apply_unsharp_mask(self, img_rgb: np.ndarray,
                            amount: Optional[float] = None) -> np.ndarray:
        """Classical unsharp-mask finishing: original + amount * (detail).

        ``amount`` defaults to the pipeline's configured ``unsharp_amount``.
        A value of 0 (or below) returns the image unchanged.
        """
        amount = float(self.unsharp_amount if amount is None else amount)
        if amount <= 0:
            return img_rgb
        blurred = cv2.GaussianBlur(img_rgb, (0, 0), float(self.unsharp_radius))
        return np.ascontiguousarray(
            cv2.addWeighted(img_rgb, 1.0 + amount, blurred, -amount, 0)
        )

    def _classical_sharpen(self, img_rgb: np.ndarray) -> np.ndarray:
        """Classical unsharp-mask sharpening fallback."""
        blurred = cv2.GaussianBlur(img_rgb, (0, 0), 3.0)
        amount = 0.5 + self.strength
        return cv2.addWeighted(img_rgb, 1 + amount, blurred, -amount, 0)

    # -- video processing ---------------------------------------------------
    def _process_video(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Process a video frame by frame."""
        cap = None
        out = None
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise InferenceError(f"Failed to open video: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise InferenceError(f"Failed to create output video: {output_path}")

            progress_callback = kwargs.get('progress_callback')
            processed = 0
            failed = 0
            temp_dir = Path(output_path).parent

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    temp_in = temp_dir / f"_sharpen_frame_{processed}.png"
                    temp_out = temp_dir / f"_sharpen_frame_{processed}_out.png"
                    cv2.imwrite(str(temp_in), frame)
                    result = self._process_image(str(temp_in), str(temp_out))
                    if result.get('success'):
                        restored = cv2.imread(str(temp_out))
                        out.write(restored)
                        processed += 1
                    else:
                        out.write(frame)
                        failed += 1
                    for p in (temp_in, temp_out):
                        try:
                            p.unlink()
                        except OSError:
                            pass
                except Exception as e:
                    logger.warning(f"Frame {processed} failed: {e}")
                    failed += 1
                    out.write(frame)

                if progress_callback:
                    progress_callback(processed / total_frames * 100)

            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': processed,
                'frames_failed': failed,
                'total_frames': total_frames,
                'model': self.model_name,
            }

        except ProcessingAborted:
            raise
        except Exception as e:
            raise InferenceError(f"Sharpen/deblur video failed: {e}")
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------
def get_sharpen_model(model_name: str = 'NAFNet', models_dir: Optional[str] = None) -> Optional[str]:
    """Return the path to a downloaded sharpen model, or None.

    ``models_dir`` defaults to the application's resolved model directory
    (``src.core.paths.models_dir()``), which is CWD-independent and points
    at the user-writable model folder in frozen builds. A plain relative
    ``"models"`` value is still honoured for backwards compatibility.
    """
    model_config = SHARPEN_MODELS.get(model_name)
    if not model_config:
        return None

    filename = model_config.get('model_file')
    if not filename:
        return None

    if models_dir is None:
        try:
            from ..core import paths as _paths
            models_dir = str(_paths.models_dir())
        except Exception:
            models_dir = "models"

    path = Path(models_dir) / filename
    if path.exists():
        return str(path)
    # Fallback: the caller (or legacy config) may rely on the repo-relative dir.
    if str(models_dir) != "models":
        legacy = Path("models") / filename
        if legacy.exists():
            return str(legacy)
    return None


def list_sharpen_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """List available sharpen/deblur models with their status."""
    results = []
    for name, config in SHARPEN_MODELS.items():
        path = get_sharpen_model(name, models_dir)
        results.append({
            'name': name,
            'description': config.get('description', ''),
            'downloaded': path is not None,
            'path': path,
        })
    return results


def sharpen_image(input_path: str, output_path: str, model_name: str = 'NAFNet',
                  device: str = 'auto', models_dir: Optional[str] = None,
                  strength: float = 1.0, overlap: Optional[int] = None,
                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Convenience function to sharpen/deblur a single image."""
    pipeline = SharpenPipeline(model_name=model_name, device=device, strength=strength,
                               overlap=overlap)
    model_path = get_sharpen_model(model_name, models_dir)
    if model_path:
        pipeline.load_model(model_path)
    else:
        logger.warning(f"No model file for {model_name}; using classical sharpening")
    return pipeline.process(input_path, output_path, progress_callback=progress_callback)
