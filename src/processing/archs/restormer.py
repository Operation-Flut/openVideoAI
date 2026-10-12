"""Restormer: Efficient Transformer for High-Resolution Image Restoration.

Implements the official ``Restormer`` architecture (S. W. Zamir et al., CVPR
2022, https://github.com/swz30/Restormer) in pure PyTorch so the single-image
defocus-deblurring checkpoint can run offline without the research repository.

The exact configuration is reconstructed 1:1 from the checkpoint weights in
``models/single_image_defocus_deblurring.pth`` (``state_dict['params']``):

* ``dim = 48``, ``num_blocks = [4, 6, 6, 8]``, ``num_refinement_blocks = 4``
* ``ffn_expansion_factor = 2.66``, ``bias = False``, ``LayerNorm_type
  = 'WithBias'``, ``num_heads = [1, 2, 4, 8]``
* The decoder's final stage (``decoder_level1``, ``refinement``) operates on
  96 channels: the level-2 upsampled tensor (48 ch) is concatenated with the
  level-1 encoder skip (48 ch), and the output head maps 96 ch back to RGB.

The net downsamples by a factor of 8 (stride-1 patch embed plus three 2x
downsamplers), so ``forward`` reflect-pads the input to a multiple of 8 and
crops the result back to the original size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Channel-wise layer norm for BCHW tensors ('WithBias' variant)."""

    def __init__(self, dim: int, layer_norm_type: str = 'WithBias'):
        super().__init__()
        if layer_norm_type == 'BiasFree':
            self.body = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.body = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return self.body(
            x.permute(0, 2, 3, 1).reshape(-1, h * w, x.shape[1])
        ).reshape(-1, h, w, x.shape[1]).permute(0, 3, 1, 2)


class Attention(nn.Module):
    """Multi-DConv-head transposed attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class FeedForward(nn.Module):
    """Gated-Dconv feed-forward network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66,
                 bias: bool = False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3,
            stride=1, padding=1, groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        gated, x = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(gated) * x
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """One Restormer transformer block (MDTA + GDFN with residual skips)."""

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float,
                 bias: bool, layer_norm_type: str):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Stem: 3x3 conv (stride 1 keeps full resolution in deblur variants)."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 48,
                 stride: int = 1, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride,
                              padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Halves resolution, doubles channels (conv + pixel unshuffle)."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Doubles resolution (conv + pixel shuffle)."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    """Restormer image-restoration network (defocus-deblurring layout).

    Parameters match the official ``single_image_defocus_deblurring.pth``
    checkpoint: ``dim=48``, ``num_blocks=[4,6,6,8]``, 4 refinement blocks,
    ``ffn_expansion_factor=2.66``, ``num_heads=[1,2,4,8]``.

    ``forward`` always preserves the input spatial size: the tensor is
    reflect-padded to a multiple of 8 (the net's total stride: a stride-1
    patch embed plus three 2x downsamplers) before the encoder and cropped
    back afterwards.
    """

    def __init__(self, dim: int = 48, num_blocks=None, num_refinement_blocks: int = 4,
                 num_heads=None, ffn_expansion_factor: float = 2.66,
                 bias: bool = False, layer_norm_type: str = 'WithBias',
                 img_channel: int = 3):
        super().__init__()
        num_blocks = num_blocks or [4, 6, 6, 8]
        num_heads = num_heads or [1, 2, 4, 8]

        self.patch_embed = OverlapPatchEmbed(img_channel, dim, stride=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=num_heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(int(dim * 2 ** 0))
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=num_heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=num_heads[3],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        # reduce_chan_level3/2 take the concatenation of the upsampled decoder
        # tensor with the matching encoder skip (192+192, 96+96).
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2 ** 2) * 2, int(dim * 2 ** 2),
            kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=num_heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2 ** 1) * 2, int(dim * 2 ** 1),
            kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_blocks[1])])

        # Final stage runs on ``dim * 2`` channels: up2_1 halves the level-2
        # channels (48) and the level-1 encoder skip (48) is concatenated
        # directly (no level-1 channel-reduction conv in this layout). Both
        # decoder_level1 and refinement use ``num_heads[0]`` (single head),
        # matching the official checkpoint weights.
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=num_heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, layer_norm_type=layer_norm_type)
            for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2 ** 1), img_channel, kernel_size=3,
                                padding=1, bias=bias)

    def _forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return inp_img + out_dec_level1

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        h, w = inp_img.shape[-2:]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
            padded = F.pad(inp_img, pad, mode='reflect')
        else:
            padded = inp_img
        out = self._forward(padded)
        if ph or pw:
            out = out[:, :, pad[2]:pad[2] + h, pad[0]:pad[0] + w]
        return out