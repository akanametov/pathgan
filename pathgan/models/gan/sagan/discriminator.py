"""Self-Attention Discriminator."""

import torch
from torch import nn, Tensor
from ..layers import Conv, ShuffleAttention


class MapDiscriminator(nn.Module):
    """MapDiscriminator.
    Parameters
    ----------
    map_channels: int, (default=3)
        Number of Map channels.
    roi_channels: int, (default=3)
        Number of ROI channels.
    hid_channels: int, (default=64)
        Number of hidden channels.
    out_channels: int, (default=1)
        Number of output (ROI) channels.
    """
    def __init__(
        self,
        map_channels: int = 3,
        roi_channels: int = 3,
        hid_channels: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.map_proj = Conv(map_channels, hid_channels//2, activation="lrelu")
        self.roi_proj = Conv(roi_channels, hid_channels//2, activation="lrelu")
        self.map_attn = ShuffleAttention(hid_channels//2)
        self.roi_attn = ShuffleAttention(hid_channels//2)

        self.dn1 = Conv(1* hid_channels, 2* hid_channels, activation="lrelu", normalization="batch")
        self.dn2 = Conv(2* hid_channels, 4* hid_channels, activation="lrelu", normalization="batch")
        self.attn = ShuffleAttention(4* hid_channels)
        self.dn3 = Conv(4* hid_channels, 8* hid_channels, activation="lrelu", normalization="batch")
        
        self.output = nn.Conv2d(8* hid_channels, out_channels, kernel_size=2)
        
    def forward(self, maps: Tensor, rois: Tensor) -> Tensor:
        mx = self.map_proj(maps)
        mx = mx + self.map_attn(mx)
        rx = self.roi_proj(rois)
        rx = rx + self.roi_attn(rx)
        fx = torch.cat([mx, rx], dim=1)

        fx = self.dn1(fx)
        fx = self.dn2(fx)
        fx = fx + self.attn(fx)
        fx = self.dn3(fx)
        fx = self.output(fx)
        return fx


class PointDiscriminator(nn.Module):
    """Point Discriminator.
    Parameters
    ----------
    point_channels: int, (default=3)
        Number of Point channels.
    roi_channels: int, (default=3)
        Number of ROI channels.
    hid_channels: int, (default=64)
        Number of hidden channels.
    out_channels: int, (default=1)
        Number of output (ROI) channels.
    """
    def __init__(
        self,
        point_channels: int = 3,
        roi_channels: int = 3,
        hid_channels: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.point_proj = Conv(point_channels, 3 * hid_channels // 4, activation="lrelu")
        self.roi_proj = Conv(roi_channels, hid_channels // 4, activation="lrelu")
        self.point_attn = ShuffleAttention(3 * hid_channels // 4)
        self.roi_attn = ShuffleAttention(hid_channels//4)

        self.dn1 = Conv(hid_channels, 2 * hid_channels, activation="lrelu", normalization="batch")
        self.dn2 = Conv(2 * hid_channels, 4 * hid_channels, activation="lrelu", normalization="batch")
        self.attn = ShuffleAttention(4 * hid_channels)
        self.dn3 = Conv(4 * hid_channels, 8 * hid_channels, activation="lrelu", normalization="batch")

        self.output = nn.Conv2d(8* hid_channels, out_channels, kernel_size=2)

    def forward(self, points: Tensor, rois: Tensor) -> Tensor:
        px = self.point_proj(points)
        px = px + self.point_attn(px)
        rx = self.roi_proj(rois)
        rx = rx + self.roi_attn(rx)
        fx = torch.cat([px, rx], dim=1)
        
        fx = self.dn1(fx)
        fx = self.dn2(fx)
        fx = fx + self.attn(fx)
        fx = self.dn3(fx)
        fx = self.output(fx)
        return fx
