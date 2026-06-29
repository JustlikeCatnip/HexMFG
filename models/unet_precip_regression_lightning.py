from models.unet_parts import Down, DoubleConv, Up, OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, ResDoubleConvDS, UpDS, UpDS_Simple, DownDS
from models.layers import CBAM
from models.regression_HexMF_UNet import Precip_regression_base
from models.regression_HexMF_GNet import Precip_regression_base_gnet
from models.regression_HexMF_GNet_aleatoric import Precip_regression_base_gnet_aleatoric
import torch
import torch.nn as nn


class HexagonalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(HexagonalConv, self).__init__()
        padding = kernel_size // 2
        self.hex_kernel = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self._init_hexagonal_weights()

    def _init_hexagonal_weights(self):

        with torch.no_grad():
            weight = self.hex_kernel.weight

            mask = torch.zeros_like(weight)
            center = weight.shape[2] // 2

            for i in range(weight.shape[2]):
                for j in range(weight.shape[3]):
                    dx = abs(i - center)
                    dy = abs(j - center)
                    if dx + dy <= center + 1 and max(dx, dy) <= center:
                        mask[:, :, i, j] = 1.0

            weight *= mask
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        return self.hex_kernel(x)


class HoneycombFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HoneycombFeatureFusion, self).__init__()
        branch_channels = out_channels // 3
        remaining_channels = out_channels - branch_channels * 2

        self.high_freq = nn.Sequential(
            HexagonalConv(in_channels, branch_channels, kernel_size=3),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.mid_freq = nn.Sequential(
            HexagonalConv(in_channels, branch_channels, kernel_size=5),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.low_freq = nn.Sequential(
            HexagonalConv(in_channels, remaining_channels, kernel_size=7),
            nn.BatchNorm2d(remaining_channels),
            nn.ReLU(inplace=True)
        )

        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.honeycomb_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        high = self.high_freq(x)
        mid = self.mid_freq(x)
        low = self.low_freq(x)

        combined = torch.cat([high, mid, low], dim=1)
        honeycomb_feats = self.honeycomb_fusion(combined)
        skip_feats = self.skip_connection(x)
        honeycomb_feats = honeycomb_feats + skip_feats

        gate_weights = self.gate(honeycomb_feats)
        gated = honeycomb_feats * gate_weights

        return gated


class PrecipMSDecomp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PrecipMSDecomp, self).__init__()
        self.honeycomb_fusion = HoneycombFeatureFusion(in_channels, out_channels)

    def forward(self, x):
        return self.honeycomb_fusion(x)


class HexMF_UNet(Precip_regression_base):
    def __init__(self, hparams):
        super(HexMF_UNet, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)

        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class HexMF_GNet(Precip_regression_base_gnet):
    def __init__(self, hparams):
        super(Precip_regression_base_gnet, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.n_masks = self.hparams.n_masks
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout

        self.precip_ms_decomp = PrecipMSDecomp(self.n_channels, 64)

        # map down
        self.inc1 = DoubleConvDS(64, 64, kernels_per_layer=kernels_per_layer)
        self.cbam11 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down11 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam12 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down12 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam13 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down13 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam14 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down14 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam15 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)

        # mask down
        self.inc2 = DoubleConvDS(self.n_masks, 64, kernels_per_layer=kernels_per_layer)
        self.cbam21 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down21 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam22 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down22 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam23 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down23 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam24 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down24 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam25 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        # up
        self.up1 = UpDS(1024 * 2, 512 * 2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512 * 2, 256 * 2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256 * 2, 128 * 2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128 * 2, 64 * 2, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64 * 2, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, m):
        x = self.precip_ms_decomp(x)

        x1 = self.inc1(x)
        x1Att = self.cbam11(x1)
        x2 = self.down11(x1)
        x2Att = self.cbam12(x2)
        x3 = self.down12(x2)
        x3Att = self.cbam13(x3)
        x4 = self.down13(x3)
        x4Att = self.cbam14(x4)
        x5 = self.down14(x4)
        x5Att = self.cbam15(x5)

        m1 = self.inc2(m)
        m1Att = self.cbam21(m1)
        m2 = self.down21(m1)
        m2Att = self.cbam22(m2)
        m3 = self.down22(m2)
        m3Att = self.cbam23(m3)
        m4 = self.down23(m3)
        m4Att = self.cbam24(m4)
        m5 = self.down24(m4)
        m5Att = self.cbam25(m5)

        x5Att = torch.cat((x5Att, m5Att), dim=1)
        x4Att = torch.cat((x4Att, m4Att), dim=1)
        x3Att = torch.cat((x3Att, m3Att), dim=1)
        x2Att = torch.cat((x2Att, m2Att), dim=1)
        x1Att = torch.cat((x1Att, m1Att), dim=1)

        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class HexMF_GNet_aleatoric(Precip_regression_base_gnet_aleatoric):
    def __init__(self, hparams):
        super(Precip_regression_base_gnet_aleatoric, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.n_masks = self.hparams.n_masks
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer
        dropout_prob = self.hparams.dropout
        self.precip_ms_decomp = PrecipMSDecomp(self.n_channels, 64)

        # map down
        self.inc1 = DoubleConvDS(64, 64, kernels_per_layer=kernels_per_layer)
        self.cbam11 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down11 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam12 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down12 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam13 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down13 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam14 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down14 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam15 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)

        # mask down
        self.inc2 = DoubleConvDS(self.n_masks, 64, kernels_per_layer=kernels_per_layer)
        self.cbam21 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down21 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam22 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down22 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam23 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down23 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam24 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down24 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam25 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        # up
        self.up1 = UpDS(1024 * 2, 512 * 2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512 * 2, 256 * 2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256 * 2, 128 * 2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128 * 2, 64 * 2, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64 * 2, self.n_classes)
        self.outc_var = OutConv(64 * 2, self.n_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, m):
        x = self.precip_ms_decomp(x)
        x1 = self.inc1(x)
        x1Att = self.cbam11(x1)
        x2 = self.down11(x1)
        x2Att = self.cbam12(x2)
        x3 = self.down12(x2)
        x3Att = self.cbam13(x3)
        x4 = self.down13(x3)
        x4Att = self.cbam14(x4)
        x5 = self.down14(x4)
        x5Att = self.cbam15(x5)
        m1 = self.inc2(m)
        m1Att = self.cbam21(m1)
        m2 = self.down21(m1)
        m2Att = self.cbam22(m2)
        m3 = self.down22(m2)
        m3Att = self.cbam23(m3)
        m4 = self.down23(m3)
        m4Att = self.cbam24(m4)
        m5 = self.down24(m4)
        m5Att = self.cbam25(m5)
        x5Att = torch.cat((x5Att, m5Att), dim=1)
        x4Att = torch.cat((x4Att, m4Att), dim=1)
        x3Att = torch.cat((x3Att, m3Att), dim=1)
        x2Att = torch.cat((x2Att, m2Att), dim=1)
        x1Att = torch.cat((x1Att, m1Att), dim=1)
        x = self.up1(x5Att, x4Att)
        x = self.dropout(x)
        x = self.up2(x, x3Att)
        x = self.dropout(x)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        var = self.outc_var(x)
        return logits, var