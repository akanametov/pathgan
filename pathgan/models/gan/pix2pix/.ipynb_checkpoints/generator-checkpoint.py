"""Generator."""

import torch
from torch import nn, Tensor
from ..layers import Conv, UpConv


class Generator(nn.Module):
    """Generator.

    Parameters
    ----------
    map_channels: int, (default=3)
        Number of Map input channels.
    point_channels: int, (default=3)
        Number of Point input channels.
    hid_channels: int, (default=32)
        Number of hidden channels.
    out_channels: int, (default=3)
        Number of output (ROI) channels.
    """
    def __init__(
        self,
        map_channels: int = 3, 
        point_channels: int = 3,
        hid_channels: int = 32,
        out_channels: int = 3,
    ):
        super().__init__()
        # input layers
        self.map_proj = Conv(map_channels, hid_channels//2, activation="lrelu")
        self.point_proj = Conv(point_channels, hid_channels//2, activation="lrelu")
        # downscale layers 
        self.dn1 = Conv(hid_channels, 2*hid_channels, normalization="instance", activation="lrelu")
        self.dn2 = Conv(2*hid_channels, 4*hid_channels, normalization="instance", activation="lrelu")
        self.dn3 = Conv(4*hid_channels, 8*hid_channels, normalization="instance", activation="lrelu")
        self.dn4 = Conv(8*hid_channels, 8*hid_channels, normalization="instance", activation="lrelu")
        # upscale layers
        self.up5 = UpConv(8*hid_channels, 8*hid_channels, normalization="instance", activation="lrelu", dropout=0.5)
        self.up4 = UpConv(16*hid_channels, 4*hid_channels, normalization="instance", activation="lrelu", dropout=0.5)
        self.up3 = UpConv(8*hid_channels, 2*hid_channels, normalization="instance", activation="lrelu")
        self.up2 = UpConv(4*hid_channels, hid_channels, normalization="instance", activation="lrelu")
        self.up1 = UpConv(2*hid_channels, 2*out_channels, normalization="instance", activation="lrelu")
        # output layer
        self.output = Conv(4*out_channels, out_channels, kernel_size=3, stride=1, activation="tanh")

    def forward(self, maps: Tensor, points: Tensor) -> Tensor:
        x = torch.cat([maps, points], 1)
        mx = self.map_proj(maps)
        px = self.point_proj(points)
        
        x1 = torch.cat([mx, px], 1)
        x2 = self.dn1(x1)
        x3 = self.dn2(x2)
        x4 = self.dn3(x3)
        x5 = self.dn4(x4)
        
        y5 = self.up5(x5)
        y4 = self.up4(torch.cat([y5, x4], 1))
        y3 = self.up3(torch.cat([y4, x3], 1))
        y2 = self.up2(torch.cat([y3, x2], 1))
        y1 = self.up1(torch.cat([y2, x1], 1))
        
        fx = self.output(torch.cat([y1, x], 1))
        return fx
