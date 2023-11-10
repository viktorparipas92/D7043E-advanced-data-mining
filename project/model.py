import torch
from torch import nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    """(convolution => [batch normalization] => ReLU activation) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_convolution(x)


class Downscaling(nn.Module):
    """Downscaling with maxpool then double convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_convolution = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolution(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_convolution(x)


class Upscaling(nn.Module):
    """Upscaling then double convolution"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            #  use the normal convolutions to reduce the number of channels
            self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True
            )
            self.convolution = DoubleConvolution(
                in_channels, out_channels, in_channels // 2
            )
        else:
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.convolution = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        pad =  [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
        ]
        x1 = F.pad(x1, pad)
        x = torch.cat([x2, x1], dim=1)
        return self.convolution(x)


class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.convolution(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvolution(n_channels, 64)
        self.down1 = Downscaling(64, 128)
        self.down2 = Downscaling(128, 256)
        self.down3 = Downscaling(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Downscaling(512, 1024 // factor)

        self.up1 = Upscaling(1024, 512 // factor, bilinear)
        self.up2 = Upscaling(512, 256 // factor, bilinear)
        self.up3 = Upscaling(256, 128 // factor, bilinear)
        self.up4 = Upscaling(128, 64, bilinear)
        self.out_convolution = OutConvolution(64, n_classes)

    def forward(self, x):
        x1 = self.in_convolution(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_convolution(x)
        return logits

    def use_checkpointing(self):
        self.in_convolution = torch.utils.checkpoint(self.in_convolution)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.out_convolution = torch.utils.checkpoint(self.out_convolution)