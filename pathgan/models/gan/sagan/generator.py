"""Self-Attention Generator."""

import torch
from torch import nn, Tensor
from ..layers import Conv, ResStage


class SAGenerator(nn.Module):
    """SAGenerator.

    Parameters
    ----------
    map_channels: int, (default=3)
        Number of Map input channels.
    point_channels: int, (default=3)
        Number of Point input channels.
    noise_channels: int, (default=1)
        Noise channels.
    hid_channels: int, (default=64)
        Number of hidden channels.
    out_channels: int, (default=3)
        Number of output (ROI) channels.
    """
    def __init__(
        self,
        map_channels: int = 3,
        point_channels: int = 3,
        noise_channels: int = 1,
        hid_channels: int = 64,
        out_channels: int = 3,
    ):
        super().__init__()
        self.map_proj = Conv(map_channels, hid_channels//4, kernel_size=3, stride=1, activation="relu")
        self.point_proj = Conv(point_channels, hid_channels//4, kernel_size=3, stride=1, activation="relu")
        self.noise_proj = Conv(noise_channels, hid_channels//2, kernel_size=3, stride=1, activation="relu")
        
        self.dn1 = ResStage(hid_channels)
        self.dn2 = ResStage(2*hid_channels, in_channels=hid_channels, upscale=False)
        self.dn3 = ResStage(4*hid_channels, in_channels=2*hid_channels, upscale=False)
        self.dn4 = ResStage(8*hid_channels, in_channels=4*hid_channels, upscale=False)

        self.up4 = ResStage(4*hid_channels, in_channels=8*hid_channels, upscale=True)
        self.up3 = ResStage(2*hid_channels, in_channels=4*hid_channels, upscale=True)
        self.up2 = ResStage(hid_channels, in_channels=2*hid_channels, upscale=True)
        self.up1 = ResStage(hid_channels//2, in_channels=hid_channels, kernel_size=3, stride=1, upscale=True)

        self.output = Conv(hid_channels//2, out_channels, kernel_size=1, stride=1, padding=0, activation="tanh")

    def forward(self, maps: Tensor, points: Tensor, noises: Tensor) -> Tensor:
        mx = self.map_proj(maps)
        px = self.point_proj(points)
        nx = self.noise_proj(noises)
        fx = torch.cat([mx, px, nx], dim=1)

        x1 = self.dn1(fx)
        x2 = self.dn2(x1)
        x3 = self.dn3(x2)
        x4 = self.dn4(x3)

        y4 = self.up4(x4)
        y3 = self.up3(y4)
        y2 = self.up2(y3)
        y1 = self.up1(y2)
        
        fx = self.output(y1)
        return fx
