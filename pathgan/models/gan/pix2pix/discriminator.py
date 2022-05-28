"""Discriminator."""

import torch
from torch import nn, Tensor
from ..layers import Conv, UpConv


class Discriminator(nn.Module):
    """Discriminator

    Parameters
    ----------
    in_channels: int, (default=9)
        Number of Point input channels.
    hid_channels: int, (default=32)
        Number of hidden channels.
    out_channels: int, (default=1)
        Number of output channels.
    """
    def __init__(
        self,
        in_channels: int = 9,
        hid_channels: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        self.dn1 = Conv(in_channels, hid_channels)
        self.dn2 = Conv(hid_channels, 2*hid_channels, activation="lrelu", normalization="batch")
        self.dn3 = Conv(2*hid_channels, 4*hid_channels, activation="lrelu", normalization="batch")
        self.dn4 = Conv(4*hid_channels, 8*hid_channels, activation="lrelu", normalization="batch")
        self.output = Conv(8*hid_channels, out_channels, kernel_size=2)
        
    def forward(self, x: Tensor) -> Tensor:
        fx = self.dn1(x)
        fx = self.dn2(fx) 
        fx = self.dn3(fx) 
        fx = self.dn4(fx) 
        fx = self.output(fx)
        return fx
