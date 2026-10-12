"""
SCUNet (Swin-Conv UNet) image-denoising architecture.

Vendored from ``cszn/SCUNet`` (``models/network_scunet.py``,
https://github.com/cszn/SCUNet) so the published
``scunet_color_real_psnr.pth`` checkpoint loads 1:1 with ``strict=True``.

Changes vs. upstream (kept structurally identical otherwise):

* ``thop``         dropped (only used by the upstream ``__main__`` demo).
* ``timm``         ``trunc_normal_`` and ``DropPath`` re-implemented locally.
* ``einops``       the ``rearrange`` calls are expressed as plain
                   ``view``/``permute``/``reshape`` operations.
* ``torch.einsum`` the two attention products use ``matmul`` instead, so the
                   model also runs on DirectML (``privateuseone``), which does
                   not implement ``einsum``.
* ``masked_fill_`` the shifted-window attention mask is applied as an additive
                   ``0``/``-inf`` float mask instead, because DirectML has no
                   ``masked_fill`` kernel (it raises ``CreateOperator``
                   failures). The SW-MSA cyclic shift likewise uses a
                   slice/``cat`` helper instead of ``torch.roll``, which
                   DirectML silently falls back to the CPU for.
* the per-block "Block Initial Type: ..." debug print is removed.

The default constructor config matches the published color real-noise
checkpoint: 7 stages x 4 Swin-conv blocks, width 64, window size 8,
``input_resolution=512``.
"""
import math

import numpy as np
import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills ``tensor`` with values from a truncated normal distribution.

    Same signature/behaviour as ``timm.models.layers.trunc_normal_``.
    """
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        # Mean too far outside the bounds; timm only logs a warning here.
        pass

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    lower = norm_cdf((a - mean) / std)
    upper = norm_cdf((b - mean) / std)
    with torch.no_grad():
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class DropPath(nn.Module):
    """Stochastic depth (``timm.models.layers.DropPath`` equivalent)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _roll_spatial(x, shift):
    """Circularly shift ``x`` (``[b, h, w, c]``) by ``shift`` along the height
    and width dims.

    Equivalent to ``torch.roll(x, shifts=(shift, shift), dims=(1, 2))`` but
    built from slicing + ``cat``. DirectML has no ``roll`` kernel and silently
    falls back to the CPU for it, which forces a device round-trip for every
    shifted-window block; this keeps the shift on-device.

    The effective shift is taken modulo each dimension's own size (like
    ``torch.roll``), so non-square feature maps are handled correctly.
    """
    if shift == 0:
        return x
    h = x.size(1)
    sh = shift % h
    if sh:
        x = torch.cat([x[:, h - sh:, :, :], x[:, :h - sh, :, :]], dim=1)
    w = x.size(2)
    sw = shift % w
    if sw:
        x = torch.cat([x[:, :, w - sw:, :], x[:, :, :w - sw, :]], dim=2)
    return x


class WMSA(nn.Module):
    """Window multi-head self-attention module (Swin Transformer style)."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # Stored as (n_heads, 2*ws-1, 2*ws-1) after the transpose dance below;
        # this exact shape matches the published checkpoints.
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads)
            .transpose(1, 2)
            .transpose(0, 1)
        )

    def generate_mask(self, h, w, p, shift):
        """Generate the SW-MSA cyclic-shift attention mask.

        Returned as an **additive** float mask (``0.0`` for valid positions,
        ``-inf`` where attention must be suppressed) so the forward pass can
        use plain tensor addition. DirectML does not implement
        ``masked_fill``/``masked_fill_`` (it raises
        ``CreateOperator`` failures), whereas addition is fully supported.
        """
        attn_mask = torch.zeros(
            h, w, p, p, p, p, dtype=self.relative_position_params.dtype,
            device=self.relative_position_params.device
        )
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = float('-inf')
        attn_mask[-1, :, s:, :, :s, :] = float('-inf')
        attn_mask[:, -1, :, :s, :, s:] = float('-inf')
        attn_mask[:, -1, :, s:, :, :s] = float('-inf')
        # rearrange('w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask.reshape(1, 1, h * w, p * p, p * p)

    def forward(self, x):
        """Forward pass; ``x`` is ``[b h w c]``, output has the same shape."""
        if self.type != 'W':
            x = _roll_spatial(x, -(self.window_size // 2))
        b, h, w, c = x.shape
        p = self.window_size
        # rearrange('b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c')
        x = x.view(b, h // p, p, w // p, p, c).permute(0, 1, 3, 2, 4, 5)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # rearrange('b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c')
        x = x.reshape(b, h_windows * w_windows, p * p, c)
        qkv = self.embedding_layer(x)
        # rearrange('b nw np (threeh c) -> threeh b nw np c', c=head_dim),
        # with threeh = 3 * n_heads (contiguous layout q,k,v per head).
        qkv = qkv.reshape(b, h_windows * w_windows, p * p, self.n_heads * 3, self.head_dim)
        q, k, v = qkv.permute(3, 0, 1, 2, 4).chunk(3, dim=0)
        # einsum('hbwpc,hbwqc->hbwpq') via matmul (DirectML-safe).
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # rearrange('h p q -> h 1 1 p q')
        sim = sim + self.relative_embedding().unsqueeze(1).unsqueeze(1)
        if self.type != 'W':
            attn_mask = self.generate_mask(
                h_windows, w_windows, self.window_size, shift=self.window_size // 2
            )
            # Additive mask (0 / -inf) instead of masked_fill_ (DirectML-safe).
            sim = sim + attn_mask.to(sim.dtype)

        probs = nn.functional.softmax(sim, dim=-1)
        # einsum('hbwij,hbwjc->hbwic') via matmul (DirectML-safe).
        output = torch.matmul(probs, v)
        # rearrange('h b w p c -> b w p (h c)')
        output = output.permute(1, 2, 3, 0, 4).reshape(
            b, h_windows * w_windows, p * p, self.n_heads * self.head_dim
        )
        output = self.linear(output)
        # rearrange('b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c')
        output = (
            output.view(b, h_windows, w_windows, p, p, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, h_windows * p, w_windows * p, c)
        )

        if self.type != 'W':
            output = _roll_spatial(output, self.window_size // 2)
        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)])
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]


class Block(nn.Module):
    """Swin-Transformer pre-norm block (window attention + MLP)."""

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    """Combined convolution + Swin-Transformer block used by SCUNet."""

    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(
            self.trans_dim, self.trans_dim, self.head_dim, self.window_size,
            self.drop_path, self.type, self.input_resolution,
        )
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = trans_x.permute(0, 2, 3, 1)  # b c h w -> b h w c
        trans_x = self.trans_block(trans_x)
        trans_x = trans_x.permute(0, 3, 1, 2)  # b h w c -> b c h w
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        return x + res


class SCUNet(nn.Module):
    """Swin-Conv UNet from cszn/SCUNet.

    The default config matches the published ``scunet_color_real_psnr.pth``
    checkpoint (7 stages x 4 blocks, width 64, window size 8).
    """

    def __init__(self, in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64, drop_path_rate=0.0, input_resolution=512):
        super().__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        # Drop-path rate for each layer.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[0])] + \
                       [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + \
                       [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + \
                       [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False)] + \
                     [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 4)
                      for i in range(config[4])]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False)] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2)
                      for i in range(config[5])]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False)] + \
                     [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution)
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)
        # self.apply(self._init_weights)

    def forward(self, x0):
        h, w = x0.size()[-2:]
        padding_bottom = int(np.ceil(h / 64) * 64 - h)
        padding_right = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x[..., :h, :w]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)