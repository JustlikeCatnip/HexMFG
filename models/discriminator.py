import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.layers import CBAM

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import PASTA


class MSST(nn.Module):
    """Multi-Scale Spatiotemporal (MSST) module for multi-scale downsampling"""

    def __init__(self, in_channels, out_channels, use_bn=True, reduction_ratio=16):
        super(MSST, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 1, 1)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.pasta = PASTA(out_channels, reduction_ratio=reduction_ratio)
        self.conv_down = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if in_channels == out_channels else \
            nn.Conv2d(out_channels, out_channels * 2, 4, 2, 1)
        if use_bn:
            self.conv_down_bn = nn.BatchNorm2d(out_channels * 2 if in_channels != out_channels else out_channels)
        self.pasta_down = PASTA(out_channels * 2 if in_channels != out_channels else out_channels,
                                reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x), 0.2)
        if self.use_bn:
            x = self.bn(x)
        x = self.pasta(x)
        x = F.leaky_relu(self.conv_down(x), 0.2)
        if self.use_bn:
            x = self.conv_down_bn(x)
        x = self.pasta_down(x)
        return x


class PAF(nn.Module):
    """Pixel Attention-oriented Fusion (PAF) module with PASTA attention"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(PAF, self).__init__()
        self.pasta = PASTA(in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        return self.pasta(x)


class PSD(nn.Module):
    """Precipitation Structure Discriminator (PSD) for final authenticity judgment"""

    def __init__(self, in_channels):
        super(PSD, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 4, 1, 1)

    def forward(self, x):
        return F.sigmoid(self.conv(x))


class HexMF_DNet(nn.Module):
    """HexMF-DNet: Discriminator component of the generative adversarial framework"""

    def __init__(self, in_channels=25, d=64):
        super(HexMF_DNet, self).__init__()

        reduction_ratio = 16

        # Stage 1: MSST - Multi-Scale Spatiotemporal encoding
        self.msst1 = nn.Sequential(
            nn.Conv2d(in_channels, d, 4, 1, 1),
            PASTA(d, reduction_ratio=reduction_ratio),
            nn.Conv2d(d, d, 4, 2, 1),
            PASTA(d, reduction_ratio=reduction_ratio)
        )

        # Stage 2: MSST with BN
        self.msst2 = nn.Sequential(
            nn.Conv2d(d, d, 4, 1, 1),
            nn.BatchNorm2d(d),
            PASTA(d, reduction_ratio=reduction_ratio),
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            PASTA(d * 2, reduction_ratio=reduction_ratio)
        )

        # Stage 3: MSST with BN
        self.msst3 = nn.Sequential(
            nn.Conv2d(d * 2, d * 2, 4, 1, 1),
            nn.BatchNorm2d(d * 2),
            PASTA(d * 2, reduction_ratio=reduction_ratio),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            PASTA(d * 4, reduction_ratio=reduction_ratio)
        )

        # Stage 4: MSST with BN
        self.msst4 = nn.Sequential(
            nn.Conv2d(d * 4, d * 4, 4, 1, 1),
            nn.BatchNorm2d(d * 4),
            PASTA(d * 4, reduction_ratio=reduction_ratio),
            nn.Conv2d(d * 4, d * 8, 4, 1, 1),
            nn.BatchNorm2d(d * 8),
            PASTA(d * 8, reduction_ratio=reduction_ratio)
        )

        # PSD module - Precipitation Structure Discriminator
        self.psd = PSD(d * 8)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)

        x = F.leaky_relu(self.msst1[0](x), 0.2)
        x = self.msst1[1](x)
        x = F.leaky_relu(self.msst1[2](x), 0.2)
        x = self.msst1[3](x)

        x = F.leaky_relu(self.msst2[1](self.msst2[0](x)), 0.2)
        x = self.msst2[2](x)
        x = F.leaky_relu(self.msst2[4](self.msst2[3](x)), 0.2)
        x = self.msst2[5](x)

        x = F.leaky_relu(self.msst3[1](self.msst3[0](x)), 0.2)
        x = self.msst3[2](x)
        x = F.leaky_relu(self.msst3[4](self.msst3[3](x)), 0.2)
        x = self.msst3[5](x)

        x = F.leaky_relu(self.msst4[1](self.msst4[0](x)), 0.2)
        x = self.msst4[2](x)
        x = F.leaky_relu(self.msst4[4](self.msst4[3](x)), 0.2)
        x = self.msst4[5](x)

        x = self.psd(x)
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()





