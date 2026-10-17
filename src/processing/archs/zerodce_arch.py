"""
Zero-DCE (Zero-Reference Deep Curve Estimation) architecture.
https://github.com/Li-Chongyi/Zero-DCE
"""
import torch
import torch.nn as nn


class DCEBlock(nn.Module):
    """Depth Curve Estimation Block."""
    def __init__(self, in_channels=3, out_channels=32):
        super(DCEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv7 = nn.Conv2d(out_channels, 3, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x7 = self.conv7(x6)
        return x7


class ZeroDCE(nn.Module):
    """Zero-Reference Deep Curve Estimation Network."""
    def __init__(self):
        super(ZeroDCE, self).__init__()
        self.model = DCEBlock()

    def forward(self, x):
        # Estimate pixel-wise curves
        curves = self.model(x)
        # Apply curves iteratively (default 8 iterations)
        enhanced_image = x
        for _ in range(8):
            enhanced_image = enhanced_image + curves * (torch.pow(enhanced_image, 2) - enhanced_image)
        return enhanced_image


__all__ = ['ZeroDCE']
