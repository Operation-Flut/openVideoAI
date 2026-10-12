"""
Deep Color Transfer architecture (pure-PyTorch inference port).

Ported from "Deep Color Transfer using Histogram Analogy" (CGI 2020,
https://github.com/codeslake/Color_Transfer_Histogram_Analogy). The released
checkpoints (`latest_net_G_A.pth` generator / histogram U-Net and
`latest_net_C_A.pth` histogram-feature encoder) run here without CUDA, on CPU
or DirectML.

The port mirrors the upstream code module-for-module:

* ``ConditionNetwork2``  - HEN: maps a 9-channel histogram cube
  (``ab_bin=64`` 2-D histogram + ``l_bin=8`` 1-D L histogram tiled) to a
  64-d colour-condition vector. ``model3`` is kept although unused, for strict
  state-dict parity.
* ``HISTUnet3_Res``      - IRN: 3-level histogram-conditioned U-Net with 3
  resnet blocks and 4 intermediate outputs (only the 5th/final output is used
  for inference, matching ``colorhistogram_model.test()``).
* Histogram extraction   - numpy ``histogramdd`` over the (a, b) and L
  channels of the skimage-LAB image, with the upstream ``rot90``/``flip``
  orientation and the zero-exclusion + imbalance-padding quirks.
* LAB <-> RGB           - exact skimage ``rgb2lab``/``lab2rgb`` (D65, 2
  degrees) implemented in numpy, so results match the upstream pipeline that
  used ``skimage.color`` directly.

Inference protocol (from ``colorhistogram_model.test()``):

* Inputs are skimage-LAB images scaled to ~[0, 1] and normalised to [-1, 1]
  (``(v - 0.5) / 0.5``), capped at 700 px on the long side.
* The content image's own histogram feature is tiled and passed as ``hist1``;
  the reference/style image's feature as ``hist2``. All three inputs are
  replicate-padded by 30 before the U-Net and symmetrically unpadded after.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# skimage-compatible LAB <-> RGB (D65, 2 degree, exact constants)
# ---------------------------------------------------------------------------
# xyz_from_rgb / inverse as defined in skimage.color.colorconv.
_XYZ_FROM_RGB = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_LAB_REF_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (float [0, 1], channels last) -> CIE-LAB, matching skimage.

    Mirror of ``skimage.color.rgb2lab`` (illuminant D65, 2-degree observer).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = rgb.copy()
    mask = linear > 0.04045
    linear[mask] = np.power((linear[mask] + 0.055) / 1.055, 2.4)
    linear[~mask] /= 12.92
    xyz = linear @ _XYZ_FROM_RGB.T
    xyz = xyz / _LAB_REF_WHITE
    f = xyz.copy()
    mask = f > 0.008856
    f[mask] = np.cbrt(f[mask])
    f[~mask] = 7.787 * f[~mask] + 16.0 / 116.0
    x, y, z = f[..., 0], f[..., 1], f[..., 2]
    lab = np.stack(
        [116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)], axis=-1
    )
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE-LAB (channels last) -> sRGB (float [0, 1]), matching skimage.

    Mirror of ``skimage.color.lab2rgb`` -> ``xyz2rgb`` (D65, 2 degrees).
    """
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0
    z = np.maximum(z, 0.0)  # skimage lab2xyz clips negative Z
    out = np.stack([x, y, z], axis=-1)
    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787
    out = out * _LAB_REF_WHITE
    rgb = out @ _RGB_FROM_XYZ.T
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * np.power(rgb[mask], 1.0 / 2.4) - 0.055
    rgb[~mask] *= 12.92
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Histogram feature encoder (HEN) - ConditionNetwork2
# ---------------------------------------------------------------------------
class ConditionNetwork2(nn.Module):
    """Histogram encoder; input ``(1, 9, 64, 64)`` -> ``(1, 64, 1, 1)``.

    Front net ``model`` reduces the 64x64 histogram cube down to a single
    pixel (four stride-2 then three stride-1 4x4 convs), then ``model2`` is a
    final Linear 64 -> 64 projection. ``model3`` is part of an unused earlier
    variant; it is kept so released state dicts load strictly.
    """

    def __init__(self, input_nc=9, output_nc=64, ngf=64, norm_layer=None,
                 use_dropout=False, padding_type='reflect'):
        super(ConditionNetwork2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        use_bias = True  # released weights were trained with InstanceNorm.
        dim2 = 128

        model = [
            nn.Conv2d(self.input_nc, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=2, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim2, dim2, kernel_size=4, padding=1, stride=1, bias=use_bias),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim2, self.output_nc, kernel_size=1, padding=0, bias=use_bias),
        ]
        self.model = nn.Sequential(*model)
        self.model2 = nn.Sequential(nn.Linear(output_nc, output_nc))
        self.model3 = nn.Sequential(nn.Linear(output_nc, 324))

    def forward(self, input_tensor):
        # Ver.2 forward used by the released checkpoints: the front net
        # already runs down to a single pixel, so no pooling is required.
        a1 = self.model(input_tensor)              # (B, 64, 1, 1)
        a2 = a1.view(a1.size(0), -1)               # (B, 64)
        a3 = self.model2(a2)                       # (B, 64)
        a3 = a3.unsqueeze(0).unsqueeze(0).permute(2, 3, 0, 1)
        return a3                                  # (1, 64, 1, 1)


# ---------------------------------------------------------------------------
# Histogram-conditioned U-Net (IRN) - HISTUnet3_Res
# ---------------------------------------------------------------------------
class _UnetEncDeep1(nn.Module):
    """Encoder head: 2x (ReLU-conv) + downsampling conv."""

    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1),
            nn.InstanceNorm2d(output_nc, affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
            nn.InstanceNorm2d(output_nc, affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(output_nc, output_nc, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(output_nc, affine=False),
        )

    def forward(self, x):
        return self.block(x)


class _UnetEncDeep2(nn.Module):
    """Encoder block: LeakyReLU -> conv -> norm (x2) + downsampling conv."""

    def __init__(self, input_nc, output_nc, use_bias):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(output_nc, affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
            nn.InstanceNorm2d(output_nc, affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(output_nc, output_nc, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(output_nc, affine=False),
        )

    def forward(self, x):
        return self.block(x)


class _UnetEnc(nn.Module):
    """Deep encoder stack: 64 -> 128 -> 256 -> 512 -> 512."""

    def __init__(self, input_channel, ngf=64, use_bias=True):
        super().__init__()
        self.netEnc1 = _UnetEncDeep1(input_channel, ngf)
        self.netEnc2 = _UnetEncDeep2(ngf * 1, ngf * 2, use_bias)
        self.netEnc3 = _UnetEncDeep2(ngf * 2, ngf * 4, use_bias)
        self.netEnc4 = _UnetEncDeep2(ngf * 4, ngf * 8, use_bias)
        self.netEnc5 = _UnetEncDeep2(ngf * 8, ngf * 8, use_bias)

    def forward(self, x):
        out1 = self.netEnc1.forward(x)
        out2 = self.netEnc2.forward(out1)
        out3 = self.netEnc3.forward(out2)
        out4 = self.netEnc4.forward(out3)
        out5 = self.netEnc5.forward(out4)
        return out1, out2, out3, out4, out5


class _UnetDecDeep2(nn.Module):
    """Decoder block: ReLU + 2x upsample + 3 conv-norm pairs."""

    def __init__(self, input_nc, output_nc, use_bias):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(output_nc, affine=False),
            nn.ReLU(True),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
            nn.InstanceNorm2d(output_nc, affine=False),
            nn.ReLU(True),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1),
            nn.InstanceNorm2d(output_nc, affine=False),
        )

    def forward(self, input1, input2, enc1, enc2, target_size):
        h, w = target_size
        input1 = F.interpolate(input1, size=(h, w), mode='bilinear', align_corners=False)
        enc1 = F.interpolate(enc1, size=(h, w), mode='bilinear', align_corners=False)
        enc2 = F.interpolate(enc2, size=(h, w), mode='bilinear', align_corners=False)
        input2 = F.interpolate(input2, size=(h, w), mode='bilinear', align_corners=False)
        return self.block(torch.cat([input1, input2, enc1, enc2], 1))


class _UnetDecDeep1(nn.Module):
    """Final decoder block (mirrors upstream's UnetDec_deep1 call, where the
    norm-layer argument lands in ``use_tanh`` and the first conv therefore has
    no bias -- exactly matching the released ``latest_net_G_A.pth`` keys)."""

    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(input_nc, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(True),
            nn.Conv2d(64, output_nc, kernel_size=3, padding=1),
        )

    def forward(self, input1, input2, enc1, enc2, target_size):
        h, w = target_size
        input1 = F.interpolate(input1, size=(h, w), mode='bilinear', align_corners=False)
        enc1 = F.interpolate(enc1, size=(h, w), mode='bilinear', align_corners=False)
        enc2 = F.interpolate(enc2, size=(h, w), mode='bilinear', align_corners=False)
        input2 = F.interpolate(input2, size=(h, w), mode='bilinear', align_corners=False)
        return self.block(torch.cat([input1, input2, enc1, enc2], 1))


class _EncResnetBlock(nn.Module):
    """Residual block that re-conditions on the two histogram features."""

    def __init__(self, dim, use_bias):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(dim * 3, affine=False),
            nn.ReLU(True),
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(dim, affine=False),
        )

    def forward(self, x, enc1, enc2):
        h, w = x.size(2), x.size(3)
        enc1 = F.interpolate(enc1, size=(h, w), mode='bilinear', align_corners=False)
        enc2 = F.interpolate(enc2, size=(h, w), mode='bilinear', align_corners=False)
        x_cat = torch.cat((x, enc1, enc2), 1)
        return x + self.conv_block(x_cat)


class HISTUnet3_Res(nn.Module):
    """Histogram-conditioned U-Net (the full ``latest_net_G_A.pth`` network)."""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, use_bias=True):
        super().__init__()
        enc_nc = 64

        # 3 -> 64 double-conv head (no bias arg -> bias=True).
        self.block_init = nn.Sequential(
            nn.Conv2d(input_nc, enc_nc, kernel_size=3, padding=1),
            nn.InstanceNorm2d(enc_nc, affine=False),
            nn.ReLU(True),
            nn.Conv2d(enc_nc, enc_nc, kernel_size=3, padding=1),
        )

        self.HistUnetEnc = _UnetEnc(input_nc, 64, use_bias)

        self.HistUnetDec1 = _UnetDecDeep2(512 + 512 + enc_nc + enc_nc, 512, use_bias)
        self.HistUnetDec2 = _UnetDecDeep2(512 + 512 + enc_nc + enc_nc, 256, use_bias)
        self.HistUnetDec3 = _UnetDecDeep2(256 + 256 + enc_nc + enc_nc, 128, use_bias)
        self.HistUnetDec4 = _UnetDecDeep2(128 + 128 + enc_nc + enc_nc, 64, use_bias)
        self.HistUnetDec5 = _UnetDecDeep1(64 + 64 + enc_nc + enc_nc, 64)

        self.ENC_Block1 = _EncResnetBlock(enc_nc, use_bias)
        self.ENC_Block2 = _EncResnetBlock(enc_nc, use_bias)
        self.ENC_Block3 = _EncResnetBlock(enc_nc, use_bias)

        self.InterOut1 = self._build_inter_out(512, output_nc)
        self.InterOut2 = self._build_inter_out(256, output_nc)
        self.InterOut3 = self._build_inter_out(128, output_nc)
        self.InterOut4 = self._build_inter_out(64, output_nc)

        self.block_last = nn.Sequential(
            nn.Conv2d(enc_nc, enc_nc, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(enc_nc, enc_nc, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(enc_nc, enc_nc, kernel_size=3, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(enc_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        )

    @staticmethod
    def _build_inter_out(dim_img, dim_out):
        return nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim_img, dim_img, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim_img, affine=False),
            nn.ReLU(True),
            nn.Conv2d(dim_img, dim_out, kernel_size=3, padding=1),
        )

    def forward(self, input_img, hist1_enc, hist2_enc):
        mid_img1, mid_img2, mid_img3, mid_img4, mid_img5 = self.HistUnetEnc(input_img)

        out_img1 = self.HistUnetDec1(mid_img5, mid_img5, hist1_enc, hist2_enc,
                                     (mid_img5.size(2), mid_img5.size(3)))
        out_img2 = self.HistUnetDec2(out_img1, mid_img4, hist1_enc, hist2_enc,
                                     (mid_img4.size(2), mid_img4.size(3)))
        out_img3 = self.HistUnetDec3(out_img2, mid_img3, hist1_enc, hist2_enc,
                                     (mid_img3.size(2), mid_img3.size(3)))
        out_img4 = self.HistUnetDec4(out_img3, mid_img2, hist1_enc, hist2_enc,
                                     (mid_img2.size(2), mid_img2.size(3)))
        out_img5 = self.HistUnetDec5(out_img4, mid_img1, hist1_enc, hist2_enc,
                                     (mid_img1.size(2), mid_img1.size(3)))
        out_img5 = F.interpolate(out_img5, size=(input_img.size(2), input_img.size(3)),
                                 mode='bilinear', align_corners=False)

        init = self.block_init(input_img)
        out_img6 = self.ENC_Block1.forward(out_img5 + init, hist1_enc, hist2_enc)
        out_img7 = self.ENC_Block2.forward(out_img6 + init, hist1_enc, hist2_enc)
        out_img = self.ENC_Block3.forward(out_img7 + init, hist1_enc, hist2_enc)

        out_img1 = self.InterOut1(out_img1)
        out_img2 = self.InterOut2(out_img2)
        out_img3 = self.InterOut3(out_img3)
        out_img4 = self.InterOut4(out_img4)
        out_img = self.block_last(out_img + init)

        return out_img1, out_img2, out_img3, out_img4, out_img


# ---------------------------------------------------------------------------
# Histogram extraction (replicates colorhistogram_model.getHistogram*)
# ---------------------------------------------------------------------------
def _histogram2d(img_tensor: torch.Tensor, num_bin: int) -> torch.Tensor:
    """(a, b) joint histogram of a (1, 3, H, W) LAB tensor -> (1, 1, 64, 64).

    ``arr`` values are in [-1, 1]; zero entries are excluded and the two
    channels rebalanced (upstream quirk), then binned as ``(v + 1) / 2``.
    """
    arr = img_tensor.detach().cpu().numpy()

    arr1 = (arr[0][1].ravel()[np.flatnonzero(arr[0][1])] + 1) / 2
    arr2 = (arr[0][2].ravel()[np.flatnonzero(arr[0][2])] + 1) / 2

    if arr1.shape[0] != arr2.shape[0]:
        if arr2.shape[0] < arr1.shape[0]:
            arr2 = np.concatenate([arr2, np.array([0.0])])
        else:
            arr1 = np.concatenate([arr1, np.array([0.0])])

    H, _edges = np.histogramdd([arr1, arr2], bins=[num_bin, num_bin],
                               range=((0, 1), (0, 1)))
    H = np.rot90(H)
    H = np.flip(H, 0)

    H_torch = torch.from_numpy(H.copy()).float().view(1, 1, num_bin, num_bin)
    total_num = float(H_torch.sum())
    if total_num > 0:
        H_torch = H_torch / total_num
    return H_torch


def _histogram1d(img_tensor: torch.Tensor, num_bin: int) -> torch.Tensor:
    """L histogram -> (1, num_bin, 1, 1), same orientation trick as upstream."""
    arr = img_tensor.detach().cpu().numpy()
    arr0 = (arr[0][0].ravel()[np.flatnonzero(arr[0][0])] + 1) / 2
    arr1 = np.zeros(arr0.size)

    H, _edges = np.histogramdd([arr0, arr1], bins=[num_bin, 1],
                               range=((0, 1), (-1, 2)))
    H_torch = torch.from_numpy(H.copy()).float().unsqueeze(0).unsqueeze(0).permute(0, 2, 1, 3)
    total_num = float(H_torch.sum())
    if total_num > 0:
        H_torch = H_torch / total_num
    return H_torch


# ---------------------------------------------------------------------------
# Full transfer
# ---------------------------------------------------------------------------
class DeepColorTransfer:
    """Runs the histogram-analogy color transfer on LAB tensors.

    ``transfer`` mirrors ``colorhistogram_model.test()`` for the non-segmented
    (``is_SR`` off) case: cap to 700 px, build histogram features, condition
    both on the input's own feature and the reference feature, and return the
    U-Net output at the capped resolution.
    """

    def __init__(self, device='cpu', ab_bin: int = 64, l_bin: int = 8,
                 max_size: int = 700, pad: int = 30):
        self.device = device
        self.ab_bin = ab_bin
        self.l_bin = l_bin
        self.max_size = max_size
        self.pad = pad
        self.irn = None
        self.hen = None

    # -- loading -------------------------------------------------------------
    def load_checkpoint(self, irn_path: str, hen_path: str) -> 'DeepColorTransfer':
        """Strict-load the released G and C checkpoints (``module.`` stripped)."""
        irn = HISTUnet3_Res(3, 3, 64, use_bias=True)
        hen = ConditionNetwork2(self.l_bin + 1, 64)

        irn_state = torch.load(irn_path, map_location='cpu', weights_only=False)
        hen_state = torch.load(hen_path, map_location='cpu', weights_only=False)
        for state in (irn_state, hen_state):
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
            # keep going with the stripped dict
            if state is irn_state and isinstance(irn_state, dict) and 'state_dict' in irn_state:
                state = {k.replace('module.', '', 1): v for k, v in irn_state['state_dict'].items()}

        irn.load_state_dict(irn_state, strict=True)
        hen.load_state_dict(hen_state, strict=True)

        self.irn = irn.eval().to(self.device)
        self.hen = hen.eval().to(self.device)
        logger.info(f"Deep Color Transfer networks loaded on {self.device}")
        return self

    # -- preprocessing -------------------------------------------------------
    @staticmethod
    def _cap_size(h: int, w: int, max_size: int):
        if h <= max_size and w <= max_size:
            return h, w
        aspect = h / w
        if h > w:
            return max_size, int(max_size / aspect)
        return int(max_size * aspect), max_size

    def _norm_input(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """BGR-free entry: expects RGB uint8, returns (1, 3, H, W) in [-1,1]."""
        lab = rgb_to_lab(rgb_uint8.astype(np.float64) / 255.0)  # skimage LAB
        lab01 = np.stack([
            lab[..., 0] / 100.0,
            (lab[..., 1] + 86.1830297444) / (98.2330538631 + 86.1830297444),
            (lab[..., 2] + 107.857300207) / (94.4781222765 + 107.857300207),
        ], axis=-1)
        tensor = torch.from_numpy(lab01.transpose(2, 0, 1)[None]).float().to(self.device)
        return (tensor - 0.5) / 0.5

    @staticmethod
    def _denorm_output(out_tensor: torch.Tensor) -> np.ndarray:
        """(1, 3, H, W) in [-1,1] -> RGB uint8 (H, W, 3)."""
        v = out_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
        lab = np.stack([
            (v[..., 0] + 1) / 2 * 100.0,
            (v[..., 1] + 1) / 2 * (98.2330538631 + 86.1830297444) - 86.1830297444,
            (v[..., 2] + 1) / 2 * (94.4781222765 + 107.857300207) - 107.857300207,
        ], axis=-1).astype(np.float64)
        rgb = lab_to_rgb(lab)
        return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)

    # -- inference -----------------------------------------------------------
    def encode_histogram(self, lab_tensor: torch.Tensor) -> torch.Tensor:
        """Histogram cube (1, 9, 64, 64) for a LAB-normalised tensor in [-1,1]."""
        hist_ab = _histogram2d(lab_tensor, self.ab_bin).to(self.device)
        hist_l = _histogram1d(lab_tensor, self.l_bin).to(self.device)
        hist_l = hist_l.repeat(1, 1, self.ab_bin, self.ab_bin)
        return torch.cat((hist_ab, hist_l), 1)

    def transfer(self, content_lab: torch.Tensor,
                 reference_hist_cube: torch.Tensor) -> torch.Tensor:
        """Run the U-Net (content already capped/conditioned by caller).

        ``content_lab``: (1, 3, H, W) in [-1,1]. ``reference_hist_cube``:
        (1, 9, 64, 64) histogram cube of the reference image.
        Returns (1, 3, H, W) in [-1,1].
        """
        if self.irn is None or self.hen is None:
            raise RuntimeError("DeepColorTransfer model not loaded")

        pad = self.pad
        hist_inp = self.encode_histogram(content_lab)
        hist_inp_feat = self.hen(hist_inp)

        hist_tar_feat = self.hen(reference_hist_cube)

        h, w = content_lab.size(2), content_lab.size(3)
        inp_padded = F.pad(content_lab, (pad, pad, pad, pad), mode='replicate')
        inp_h = hist_inp_feat.repeat(1, 1, h, w)
        tar_h = hist_tar_feat.repeat(1, 1, h, w)
        inp_h = F.pad(inp_h, (pad, pad, pad, pad), mode='replicate')
        tar_h = F.pad(tar_h, (pad, pad, pad, pad), mode='replicate')

        with torch.no_grad():
            _, _, _, _, out = self.irn(inp_padded, inp_h, tar_h)

        return out[:, :, pad:out.size(2) - pad, pad:out.size(3) - pad]

    def process(self, content_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
        """Apply the reference grade to a content image (both RGB uint8)."""
        if self.irn is None or self.hen is None:
            raise RuntimeError("DeepColorTransfer model not loaded")

        orig_h, orig_w = content_rgb.shape[:2]
        ch, cw = self._cap_size(orig_h, orig_w, self.max_size)
        rh, rw = self._cap_size(*reference_rgb.shape[:2], self.max_size)

        content_small = _resize_rgb(content_rgb, cw, ch)
        reference_small = _resize_rgb(reference_rgb, rw, rh)

        content_lab = self._norm_input(content_small)
        reference_hist_cube = None
        if self.hen is not None:
            # Pre-encode the reference histogram once.
            reference_hist_cube = self.encode_histogram(self._norm_input(reference_small))

        out_lab = self.transfer(content_lab, reference_hist_cube)
        rgb_out = self._denorm_output(out_lab)

        if (rgb_out.shape[0], rgb_out.shape[1]) != (orig_h, orig_w):
            rgb_out = _resize_rgb(rgb_out, orig_w, orig_h)
        return rgb_out


def _resize_rgb(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def create_deep_color_transfer(irn_path: str, hen_path: str,
                               device: str = 'cpu') -> DeepColorTransfer:
    """Build and load the two released checkpoints into a ready model."""
    dct = DeepColorTransfer(device=device)
    return dct.load_checkpoint(irn_path, hen_path)