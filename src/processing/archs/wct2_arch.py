"""
WCT2 (Whitening and Coloring Transform) architecture.

Vendored, self-contained copy of the official implementation so the project
does not depend on the upstream repository at runtime. Only the pieces needed
for reference-based color transfer are kept: the wavelet encoder/decoder and
the WCT core. The optional segmentation path (FastPhotoStyle's label masks) is
dropped because "Match Color" has no masks, which means ``feature_wct`` always
takes the plain ``wct_core`` branch.

Licensing / attribution
-----------------------
* ``model.py`` and ``transfer.py`` from https://github.com/clovaai/WCT2
  Copyright (c) 2019 NAVER Corp. -- MIT License.
* The WCT core (``svd``/``wct_core``) derives from ``photo_wct.py`` of
  https://github.com/NVIDIA/FastPhotoStyle
  Copyright (C) 2018 NVIDIA Corporation -- CC BY-NC-SA 4.0.

Only the ``svd`` helper was modernised: ``torch.svd`` is deprecated, so it now
uses ``torch.linalg.svd`` (which lands on the CPU for DirectML builds).
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Wavelet encoder / decoder (from clovaai/WCT2 model.py, MIT)
# ---------------------------------------------------------------------------

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    # NOTE: .clone() keeps the expanded Haar filters contiguous. Without it the
    # stride-0 expanded tensor is not a valid load_state_dict() destination on
    # recent PyTorch ("more than one element ... single memory location"). The
    # values are identical, so this does not change the numerics.
    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError


class WaveEncoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveEncoder, self).__init__()
        self.option_unpool = option_unpool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}
        if self.option_unpool == 'sum':
            if level == 1:
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                out = self.relu(self.conv1_2(self.pad(out)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                return LL
            elif level == 2:
                out = self.relu(self.conv2_1(self.pad(x)))
                out = self.relu(self.conv2_2(self.pad(out)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                return LL
            elif level == 3:
                out = self.relu(self.conv3_1(self.pad(x)))
                out = self.relu(self.conv3_2(self.pad(out)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                return LL
            else:
                return self.relu(self.conv4_1(self.pad(x)))

        elif self.option_unpool == 'cat5':
            if level == 1:
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                return out

            elif level == 2:
                out = self.relu(self.conv1_2(self.pad(x)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                out = self.relu(self.conv2_1(self.pad(LL)))
                return out

            elif level == 3:
                out = self.relu(self.conv2_2(self.pad(x)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                out = self.relu(self.conv3_1(self.pad(LL)))
                return out

            else:
                out = self.relu(self.conv3_2(self.pad(x)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                out = self.relu(self.conv4_1(self.pad(LL)))
                return out
        else:
            raise NotImplementedError


class WaveDecoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveDecoder, self).__init__()
        self.option_unpool = option_unpool

        if option_unpool == 'sum':
            multiply_in = 1
        elif option_unpool == 'cat5':
            multiply_in = 5
        else:
            raise NotImplementedError

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.recon_block3 = WaveUnpool(256, option_unpool)
        if option_unpool == 'sum':
            self.conv3_4 = nn.Conv2d(256 * multiply_in, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(256 * multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = WaveUnpool(128, option_unpool)
        if option_unpool == 'sum':
            self.conv2_2 = nn.Conv2d(128 * multiply_in, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128 * multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64, option_unpool)
        if option_unpool == 'sum':
            self.conv1_2 = nn.Conv2d(64 * multiply_in, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64 * multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            LH, HL, HH = skips['pool3']
            original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            out = self.recon_block3(out, LH, HL, HH, original)
            _conv3_4 = self.conv3_4 if self.option_unpool == 'sum' else self.conv3_4_2
            out = self.relu(_conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            LH, HL, HH = skips['pool2']
            original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            out = self.recon_block2(out, LH, HL, HH, original)
            _conv2_2 = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            LH, HL, HH = skips['pool1']
            original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            out = self.recon_block1(out, LH, HL, HH, original)
            _conv1_2 = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            return self.conv1_1(self.pad(x))


# ---------------------------------------------------------------------------
# WCT core (from NVIDIA/FastPhotoStyle, CC BY-NC-SA 4.0)
# ---------------------------------------------------------------------------

def svd(feat, iden=False, device='cpu'):
    size = feat.size()
    mean = torch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = torch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = torch.mm(_feat, _feat.t())
    if iden:
        conv += torch.eye(size[0], device=device)
    # torch.svd is deprecated; torch.linalg.svd returns Vh, so transpose it back
    # into the V layout the original code expects.
    u, e, vh = torch.linalg.svd(conv, full_matrices=True)
    return u, e, vh.transpose(-2, -1)


def get_squeeze_feat(feat):
    _feat = feat.squeeze(0)
    size = _feat.size(0)
    return _feat.view(size, -1).clone()


def get_rank(singular_values, dim, eps=0.00001):
    r = dim
    for i in range(dim - 1, -1, -1):
        if singular_values[i] >= eps:
            r = i + 1
            break
    return r


def wct_core(cont_feat, styl_feat, weight=1, device='cpu'):
    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    _, c_e, c_v = svd(cont_feat, iden=True, device=device)

    styl_feat = get_squeeze_feat(styl_feat)
    s_mean = torch.mean(styl_feat, 1)
    _, s_e, s_v = svd(styl_feat, iden=True, device=device)
    k_s = get_rank(s_e, styl_feat.size()[0])
    s_d = (s_e[0:k_s]).pow(0.5)
    EDE = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d) * weight), (s_v[:, 0:k_s].t()))

    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cont_feat)

    target_feature = torch.mm(EDE, whiten_cF)
    target_feature = target_feature + s_mean.unsqueeze(1).expand_as(target_feature)
    target_feature.clamp_(cont_min, cont_max)

    return target_feature


def feature_wct(content_feat, style_feat, weight=1, alpha=1, device='cpu'):
    """Colour-transfer one feature map (no segmentation)."""
    target_feature = wct_core(content_feat, style_feat, weight=weight, device=device)
    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat
    return target_feature


# ---------------------------------------------------------------------------
# WCT2 orchestration (from clovaai/WCT2 transfer.py, MIT)
# ---------------------------------------------------------------------------

class WCT2(nn.Module):
    """Reference-based color/style transfer using the wavelet WCT2 network."""

    def __init__(self, transfer_at=('encoder', 'decoder', 'skip'),
                 option_unpool='cat5', device='cpu'):
        super().__init__()
        self.transfer_at = set(transfer_at)
        if self.transfer_at - {'encoder', 'decoder', 'skip'}:
            raise ValueError(f"invalid transfer_at: {transfer_at}")
        if not self.transfer_at:
            raise ValueError("empty transfer_at")
        self.device = device
        self.encoder = WaveEncoder(option_unpool)
        self.decoder = WaveDecoder(option_unpool)
        self._loaded = False

    def load_weights(self, encoder_path: str, decoder_path: str, device: str = None) -> None:
        """Load the encoder/decoder checkpoints and move the net onto the device."""
        if device is not None:
            self.device = device
        state = torch.load(encoder_path, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(state, strict=True)
        state = torch.load(decoder_path, map_location='cpu', weights_only=False)
        self.decoder.load_state_dict(state, strict=True)
        self.to(self.device)
        self.eval()
        self._loaded = True

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def _get_all_features(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def encode_style(self, style):
        """Encode a style image once so it can be reused for many frames."""
        if not self._loaded:
            raise RuntimeError("WCT2 weights have not been loaded")
        return self._get_all_features(style.to(self.device))

    def transfer(self, content, style, alpha: float = 1.0):
        """Transfer the style/color of ``style`` onto ``content`` (both BCHW)."""
        style_feats, style_skips = self.encode_style(style)
        return self.transfer_with_style(content, style_feats, style_skips, alpha=alpha)

    def transfer_with_style(self, content, style_feats, style_skips, alpha: float = 1.0):
        """Like :meth:`transfer`, but reuses precomputed style features."""
        if not self._loaded:
            raise RuntimeError("WCT2 weights have not been loaded")

        content = content.to(self.device)

        content_feat = content
        content_skips = {}

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(
                    content_feat, style_feats['encoder'][level],
                    alpha=alpha, device=self.device)

        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(
                        content_skips[skip_level][component],
                        style_skips[skip_level][component],
                        alpha=alpha, device=self.device)

        for level in [4, 3, 2, 1]:
            if ('decoder' in self.transfer_at and level in style_feats['decoder']
                    and level in wct2_dec_level):
                content_feat = feature_wct(
                    content_feat, style_feats['decoder'][level],
                    alpha=alpha, device=self.device)
            content_feat = self.decode(content_feat, content_skips, level)

        return content_feat
