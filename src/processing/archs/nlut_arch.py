"""
NLUT architecture (pure-PyTorch inference port).

Ports the neural 3D-LUT generator from the NLUT project
(https://github.com/semchan/NLUT) so its checkpoints can run inside
openVideoAI without the CUDA ``trilinear`` extension.

The port mirrors the upstream code module-for-module:

* ``vgg`` / ``Net``    - fixed VGG-19 feature encoder sliced into
                         ``enc_1..enc_4`` (relu1_1 .. relu4_1).
* ``SplattingBlock2``  - AdaIN fusion of content/style features.
* ``CLUT`` (mode "++", n=2048 / s=32 / w=32 / dim=33) with
  ``cube_to_lut`` basis reconstruction.
* ``NLUTNet``          - predicts per-image basis weights, reconstructs a
  3D LUT and applies it with a pure-PyTorch version of the CUDA
  ``trilinear`` kernel (``apply_trilinear_lut``).

Checkpoint layout: ``torch.save({'iter':.., 'state_dict':.., 'optimizer':..})``
(training checkpoints such as ``336999_style_lut.pth``). The encoder weights
live inside the checkpoint, so ``models/vgg_normalised.pth`` is *not* required.

The LUT coordinates follow the CUDA kernel exactly:
``binsize = 1.000001/(dim-1)``, ``id = r_id + g_id*dim + b_id*dim*dim``,
trilinear weights from ``fmod(v,binsize)/binsize``. LUT memory layout is
``[channel][b][g][r]`` (channel offset ``dim**3``).
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = True

try:
    torch  # noqa: B018
except ImportError:  # pragma: no cover - torch is a hard runtime dependency
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# VGG feature encoder (verbatim from NLUT net.py)
# ---------------------------------------------------------------------------
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, last layer used by the encoder
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)


# TensorFlow-style ImageNet normalisation used by NLUTNet.
_PRE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_PRE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - _PRE_MEAN.to(x.device)) / _PRE_STD.to(x.device)


class Net(nn.Module):
    """VGG encoder with fixed, frozen slices (input -> relu1_1..relu4_1)."""

    def __init__(self, encoder: nn.Sequential):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])   # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input: torch.Tensor):
        """Return [relu1_1, relu2_1, relu3_1, relu4_1]."""
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content: torch.Tensor, style: torch.Tensor):
        assert content.size()[:2] == style.size()[:2]
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        nn.init.normal_(self.conv2d.weight, mean=0, std=0.5)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.bn.weight, mean=0, std=0.5)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.bn(out)
        return out


class SplattingBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SplattingBlock2, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3, 1)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1)
        self.adain = AdaIN()

    def forward(self, c, s):
        c1 = torch.tanh(self.conv1(c))
        c = torch.tanh(self.conv2(c1 + c))
        s1 = torch.tanh(self.conv1(s))
        s = torch.tanh(self.conv2(s1 + s))
        return self.adain(c, s)


# ---------------------------------------------------------------------------
# LUT container
# ---------------------------------------------------------------------------
def cube_to_lut(cube: torch.Tensor) -> torch.Tensor:
    """Rearrange a cube ``(n,)3,d,d,d`` into the kernel's ``[c][b][g][r]`` flat order.

    Ported verbatim from NLUT utils/LUT.py.
    """
    if len(cube.shape) == 5:
        to_shape = [[0, 2, 3, 1], [0, 2, 1, 3]]
    else:
        to_shape = [[1, 2, 0], [1, 0, 2]]
    lut = torch.empty_like(cube)
    lut[..., 0, :, :, :] = cube[..., 0, :, :, :].permute(*to_shape[0])
    lut[..., 1, :, :, :] = cube[..., 1, :, :, :].permute(*to_shape[1])
    lut[..., 2, :, :, :] = cube[..., 2, :, :, :]
    return lut


class CLUT(nn.Module):
    """Composable / compressed 3D LUTs (mode ``++`` used by the released model)."""

    def __init__(self, num, dim=33, s=-1, w=-1, *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s, self.w = s, w
        # +: compressed; -: uncompressed
        if s == -1 and w == -1:  # standard 3DLUT
            self.mode = '--'
            self.LUTs = nn.Parameter(torch.zeros(num, 3, dim, dim, dim))
        elif s != -1 and w == -1:
            self.mode = '+-'
            self.s_Layers = nn.Parameter(torch.rand(dim, s) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(s, num * 3 * dim * dim))
        elif s == -1 and w != -1:
            self.mode = '-+'
            self.w_Layers = nn.Parameter(torch.rand(w, dim * dim) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(num * 3 * dim, w))
        else:  # full-version CLUT
            self.mode = '++'
            self.s_Layers = nn.Parameter(torch.rand(dim, s) / 5 - 0.1)
            self.w_Layers = nn.Parameter(torch.rand(w, dim * dim) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(s * num * 3, w))

    def reconstruct_luts(self):
        dim = self.dim
        num = self.num
        if self.mode == '--':
            D3LUTs = self.LUTs
        else:
            if self.mode == '+-':
                CUBEs = (self.s_Layers.mm(self.LUTs).reshape(dim, num * 3, dim * dim)
                         .permute(1, 0, 2).reshape(num, 3, self.dim, self.dim, self.dim))
            if self.mode == '-+':
                CUBEs = self.LUTs.mm(self.w_Layers).reshape(num, 3, self.dim, self.dim, self.dim)
            if self.mode == '++':
                CUBEs = (self.s_Layers.mm(
                    self.LUTs.mm(self.w_Layers).reshape(-1, num * 3 * dim * dim))
                    .reshape(dim, num * 3, dim ** 2).permute(1, 0, 2)
                    .reshape(num, 3, self.dim, self.dim, self.dim))
            D3LUTs = cube_to_lut(CUBEs)
        return D3LUTs

    def combine(self, weight: torch.Tensor, TVMN=None):
        dim = self.dim
        num = self.num
        D3LUTs = self.reconstruct_luts()
        if TVMN is None:
            tvmn = 0
        else:
            tvmn = TVMN(D3LUTs)
        D3LUT = weight.mm(D3LUTs.reshape(num, -1)).reshape(-1, 3, dim, dim, dim)
        return D3LUT, tvmn

    def forward(self, weight: torch.Tensor, TVMN=None):
        lut, tvmn = self.combine(weight, TVMN)
        return lut, tvmn


class TVMN(nn.Module):
    """Total-variation / monotonicity regulariser (training only, kept for parity)."""

    def __init__(self, dim=33):
        super(TVMN, self).__init__()
        self.dim = dim
        self.relu = nn.ReLU()

        weight_r = torch.ones(1, 1, dim, dim, dim - 1, dtype=torch.float32)
        weight_r[..., (0, dim - 2)] *= 2.0
        weight_g = torch.ones(1, 1, dim, dim - 1, dim, dtype=torch.float32)
        weight_g[..., (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(1, 1, dim - 1, dim, dim, dtype=torch.float32)
        weight_b[..., (0, dim - 2), :, :] *= 2.0
        self.register_buffer('weight_r', weight_r, persistent=False)
        self.register_buffer('weight_g', weight_g, persistent=False)
        self.register_buffer('weight_b', weight_b, persistent=False)
        self.register_buffer('tvmn_shape', torch.empty(3), persistent=False)

    def forward(self, LUT):
        dim = self.dim
        tvmn = 0 + self.tvmn_shape
        if len(LUT.shape) > 3:  # n,3,d,d,d or 3,d,d,d
            dif_r = LUT[..., :-1] - LUT[..., 1:]
            dif_g = LUT[..., :-1, :] - LUT[..., 1:, :]
            dif_b = LUT[..., :-1, :, :] - LUT[..., 1:, :, :]
            tvmn[0] = (torch.mean(dif_r ** 2 * self.weight_r[:, 0]) +
                       torch.mean(dif_g ** 2 * self.weight_g[:, 0]) +
                       torch.mean(dif_b ** 2 * self.weight_b[:, 0]))
            tvmn[1] = (torch.mean(self.relu(dif_r * self.weight_r[:, 0]) ** 2) +
                       torch.mean(self.relu(dif_g * self.weight_g[:, 0]) ** 2) +
                       torch.mean(self.relu(dif_b * self.weight_b[:, 0]) ** 2))
            tvmn[2] = 0
        else:  # n,3,d or 3,d
            dif = LUT[..., :-1] - LUT[..., 1:]
            tvmn[1] = torch.mean(self.relu(dif))
            dif = dif ** 2
            dif[..., (0, dim - 2)] *= 2.0
            tvmn[0] = torch.mean(dif)
            tvmn[2] = 0
        return tvmn


# ---------------------------------------------------------------------------
# Trilinear interpolation (pure-PyTorch replacement for the CUDA kernel)
# ---------------------------------------------------------------------------
def apply_trilinear_lut(lut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply a 3D LUT with trilinear sampling, matching the NLUT CUDA kernel.

    ``lut`` is ``(n, 3, D, D, D)`` with the ``[channel][b][g][r]`` layout the
    kernel expects (see :func:`cube_to_lut`). ``x`` is ``(B, 3, H, W)`` RGB in
    ``[0, 1]``. When ``lut`` has a single entry it is applied to every image;
    otherwise ``lut.shape[0]`` must equal ``B`` (one LUT per batch item).

    The binning/index math is byte-for-byte the CUDA forward kernel:
    ``binsize = 1.000001/(D-1)``, ``id = r_id + g_id*D + b_id*D*D`` and the
    fractional offsets are ``fmod(v, binsize)/binsize`` (non-negative for
    clamped inputs, so plain subtraction is equivalent).
    """
    D = lut.size(-1)
    binsize = 1.000001 / (D - 1)
    B, C, H, W = x.shape
    n_lut = lut.size(0)

    x = x.clamp(0.0, 1.0)
    xf = x.reshape(B, C, -1)  # (B, 3, HW)
    r = xf[:, 0]
    g = xf[:, 1]
    b = xf[:, 2]

    r_id = torch.floor(r / binsize).long().clamp_(0, D - 2)
    g_id = torch.floor(g / binsize).long().clamp_(0, D - 2)
    b_id = torch.floor(b / binsize).long().clamp_(0, D - 2)

    r_d = (r - r_id.float() * binsize) / binsize
    g_d = (g - g_id.float() * binsize) / binsize
    b_d = (b - b_id.float() * binsize) / binsize

    base = r_id + g_id * D + b_id * (D * D)  # (B, HW)
    ids = torch.stack([
        base,
        base + 1,
        base + D,
        base + D + 1,
        base + D * D,
        base + D * D + 1,
        base + D * D + D,
        base + D * D + D + 1,
    ], dim=0)  # (8, B, HW)

    w = torch.stack([
        (1 - r_d) * (1 - g_d) * (1 - b_d),
        r_d * (1 - g_d) * (1 - b_d),
        (1 - r_d) * g_d * (1 - b_d),
        r_d * g_d * (1 - b_d),
        (1 - r_d) * (1 - g_d) * b_d,
        r_d * (1 - g_d) * b_d,
        (1 - r_d) * g_d * b_d,
        r_d * g_d * b_d,
    ], dim=0)  # (8, B, HW)

    if n_lut == 1:
        lut3 = lut.reshape(3, D * D * D)
        acc = None
        for corner in range(8):
            idx = ids[corner].reshape(1, -1).expand(3, -1).contiguous()  # (3, B*HW)
            pick = lut3.gather(1, idx)  # (3, B*HW)
            pick = pick * w[corner].reshape(1, -1)
            acc = pick if acc is None else acc + pick
        out = acc  # (3, B*HW)
        out = out.transpose(0, 1).reshape(B, HW := H * W, 3).permute(0, 2, 1).reshape(B, C, H, W)
        return out

    # One LUT per batch item.
    res = torch.empty_like(x)
    for i in range(n_lut):
        res[i:i + 1] = apply_trilinear_lut(lut[i:i + 1], x[i:i + 1])
    return res


class TrilinearInterpolation(nn.Module):
    """Module wrapper kept for state-dict parity with upstream NLUTNet."""

    def __init__(self, mo=False, clip=False):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return apply_trilinear_lut(lut, x)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
class NLUTNet(nn.Module):
    """Image-adaptive 3D LUT generator (released ``2048+32+32`` / ``dim 33``)."""

    def __init__(self, nsw, dim, *args, **kwargs):
        super(NLUTNet, self).__init__()
        self.encoder = Net(vgg)
        self.encoder.eval()
        self.adain = AdaIN()

        self.SB2 = SplattingBlock2(64, 256)
        self.SB3 = SplattingBlock2(128, 256)
        self.SB4 = SplattingBlock2(256, 256)
        self.SB5 = SplattingBlock2(512, 256)

        self.pg5 = nn.AdaptiveAvgPool2d(3)
        self.pg4 = nn.AdaptiveAvgPool2d(3)
        self.pg3 = nn.AdaptiveAvgPool2d(3)
        self.pg2 = nn.AdaptiveAvgPool2d(3)

        nsw_split = nsw.split('+')
        num, s, w = int(nsw_split[0]), int(nsw_split[1]), int(nsw_split[2])
        self.CLUTs = CLUT(num, dim, s, w)
        self.TrilinearInterpolation = TrilinearInterpolation()
        last_channel = 256 * 4
        self.classifier = nn.Sequential(
            nn.Conv2d(last_channel, 512, 3, 2),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.Conv2d(512, 512 * 2, 1, 1),
            nn.BatchNorm2d(512 * 2),
            nn.Tanh(),
            nn.Conv2d(512 * 2, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.Conv2d(512, num, 1, 1),
            nn.BatchNorm2d(num),
        )

    def forward(self, img, img_org, style, TVMN=None):
        content = _normalize(img)
        style = _normalize(style)

        # Features are computed on 256x256 crops of content and style.
        resize_style = F.interpolate(style, (256, 256), mode='bilinear', align_corners=False)
        resize_content = F.interpolate(content, (256, 256), mode='bilinear', align_corners=False)
        style_feats = self.encoder.encode_with_intermediate(resize_style)
        content_feat = self.encoder.encode_with_intermediate(resize_content)

        stylized5 = self.SB5(content_feat[-1], style_feats[-1])
        stylized4 = self.SB4(content_feat[-2], style_feats[-2])
        stylized3 = self.SB3(content_feat[-3], style_feats[-3])
        stylized2 = self.SB2(content_feat[-4], style_feats[-4])

        stylized5 = self.pg5(stylized5)
        stylized4 = self.pg4(stylized4)
        stylized3 = self.pg3(stylized3)
        stylized2 = self.pg2(stylized2)

        stylized1 = torch.cat((stylized2, stylized3, stylized4, stylized5), dim=1)
        pred = self.classifier(stylized1)[:, :, 0, 0]

        D3LUT, tvmn = self.CLUTs(pred, TVMN)

        img_out = self.TrilinearInterpolation(D3LUT, img_org)
        img_out = img_out + img_org

        return img_out, img_out, {"LUT": D3LUT, "tvmn": tvmn}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, nsw: str = '2048+32+32',
                        dim: int = 33, device: str = 'cpu') -> 'NLUTNet':
        """Build NLUTNet and load a NLUT training checkpoint.

        ``checkpoint_path`` points at a ``{iter, state_dict, optimizer}``
        checkpoint (e.g. ``336999_style_lut.pth``). The encoder weights are
        part of the checkpoint, so the separate VGG file is not loaded.
        """
        model = cls(nsw, dim)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise ValueError(
                "NLUT checkpoint state dict mismatch.\n"
                f"missing: {missing}\nunexpected: {unexpected}"
            )
        model.eval()
        return model.to(device)