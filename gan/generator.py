import torch
from torch import nn
from .modules import ConvReLU, ConvTanh, ConvBnReLU, ConvBnReLU
from .modules import ResidualStage, ConvResidualStage, UpConvResidualStage

class Generator(nn.Module):
    '''
    The ROI Generator

    Args:
        map_channels (default: int=3): Number of Map input channels 
        point_channels (default: int=3): Number of Point input channels 
        noise_channels (default: int=1): Number of Noise input channels
        hid_channels (default: int=64): Number of hidden channels
        out_channels (default: int=3): Number of output (ROI) channels
    '''
    def __init__(self,
                 map_channels=3,
                 point_channels=3,
                 noise_channels=1,
                 hid_channels=64,
                 out_channels=3):
        super().__init__()
        self.InputMap=ConvReLU(map_channels, hid_channels//4, kernel_size=3, stride=1)
        self.InputPoint=ConvReLU(point_channels, hid_channels//4, kernel_size=3, stride=1)
        self.InputNoise=ConvReLU(noise_channels, hid_channels//2, kernel_size=3, stride=1)
        
        self.DownBlock0 = ResidualStage(hid_channels)
        
        self.DownBlock1 = ConvResidualStage(1* hid_channels, 2* hid_channels)
        self.DownBlock2 = ConvResidualStage(2* hid_channels, 4* hid_channels)
        self.DownBlock3 = ConvResidualStage(4* hid_channels, 8* hid_channels)

        self.UpBlock3 = UpConvResidualStage(8* hid_channels, 4* hid_channels)
        self.UpBlock2 = UpConvResidualStage(4* hid_channels, 2* hid_channels)
        self.UpBlock1 = UpConvResidualStage(2* hid_channels, 1* hid_channels)
        
        self.UpBlock0  =  ConvResidualStage(2* hid_channels, hid_channels//2, kernel_size=3, stride=1)
        
        self.Output=ConvTanh(hid_channels//2, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, maps, points, noises):
        m = self.InputMap(maps)
        p = self.InputPoint(points)
        n = self.InputNoise(noises)
        mpn = torch.cat([m, p, n], dim=1)
        
        d0 = self.DownBlock0(mpn)
        d1 = self.DownBlock1(d0)
        d2 = self.DownBlock2(d1)
        d3 = self.DownBlock3(d2)
        
        up3 = self.UpBlock3(d3)
        up2 = self.UpBlock2(up3)
        up1 = self.UpBlock1(up2)
        up0 = self.UpBlock0(torch.cat([up1, d0], dim=1))
        
        out = self.Output(up0)
        return out
