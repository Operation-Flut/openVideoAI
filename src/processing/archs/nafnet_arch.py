"""NAFNet (Nonlinear Activation Free Network) architecture.

Vendored for the catalog's NAFNet-SIDD / NAFNet-GoPro / NAFNet-REDS entries.
The block layout differs per checkpoint (SIDD uses a wider encoder/middle stack
than GoPro/REDS), so it is taken from the catalog's ``runtime`` block for the
selected model rather than hardcoded here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Pad to a multiple of 2**depth so every downsample/upsample pair lines
        # up exactly (4 levels -> 16).
        self.padder_size = 2 ** len(enc_blk_nums)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h or mod_pad_w:
            # DirectML's 'reflect' padding applies the height/width pad amounts
            # to the wrong spatial axis for non-square inputs, which breaks the
            # encoder/decoder skip-connection shapes. 'replicate' is implemented
            # correctly there and is a safe, seam-free alternative.
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


# Block layouts of the released width-64 checkpoints. The catalog's ``runtime``
# block overrides these, so a new variant only needs a catalog entry.
SIDD_ARCH_CONFIG = {
    'width': 64,
    'middle_blk_num': 12,
    'enc_blk_nums': [2, 2, 4, 8],
    'dec_blk_nums': [2, 2, 2, 2],
}

GOPRO_ARCH_CONFIG = {
    'width': 64,
    'middle_blk_num': 1,
    'enc_blk_nums': [1, 1, 1, 28],
    'dec_blk_nums': [1, 1, 1, 1],
}


def from_runtime_config(runtime_config: dict, default: dict = None) -> NAFNet:
    """Build a NAFNet from a catalog ``runtime`` block (snake_case keys)."""
    config = dict(default or GOPRO_ARCH_CONFIG)
    for key in ('width', 'middle_blk_num', 'enc_blk_nums', 'dec_blk_nums', 'img_channel'):
        if runtime_config.get(key) is not None:
            config[key] = runtime_config[key]
    return NAFNet(
        img_channel=int(config.get('img_channel', 3)),
        width=int(config['width']),
        middle_blk_num=int(config['middle_blk_num']),
        enc_blk_nums=list(config['enc_blk_nums']),
        dec_blk_nums=list(config['dec_blk_nums']),
    )
