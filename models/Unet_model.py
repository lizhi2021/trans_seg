import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetEncoder(nn.Module):  # Basic Unet of Modality weighted Unet

    def __init__(self, n_channels, bilinear=True):
        super(UNetEncoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class FusionBlock(nn.Module):  # fusion block for MS_UNet
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=1,
                stride=1)

    def forward(self, x_pv, x_art):
        x = torch.cat([x_pv, x_art], 1)
        return self.conv1x1(x)

class MS_UNet(nn.Module):
    def __init__(self, num_classes, num_channels, bilinear=True):
        super().__init__()
        self.PV_encoder = UNetEncoder(num_channels)
        self.ART_encoder = UNetEncoder(num_channels)

        # 1*1 conv
        self.cf1 = FusionBlock(128, 64)
        self.cf2 = FusionBlock(256, 128)
        self.cf3 = FusionBlock(512, 256)
        self.cf4 = FusionBlock(1024, 512)
        self.cf5 = FusionBlock(1024, 512)

        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)


    def forward(self, x_pv, x_art):
        x1_pv, x2_pv, x3_pv, x4_pv, x5_pv = self.PV_encoder(x_pv)
        x1_art, x2_art, x3_art, x4_art, x5_art = self.ART_encoder(x_art)

        x1 = self.cf1(x1_pv, x1_art)      # with conv1*1
        x2 = self.cf2(x2_pv, x2_art)
        x3 = self.cf3(x3_pv, x3_art)
        x4 = self.cf4(x4_pv, x4_art)
        x5 = self.cf5(x5_pv, x5_art)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



