import torch
from torch import nn
from .modules import ConvReLU, ConvTanh, ConvBnReLU, ConvBnReLU
from .modules import ResidualStage, ConvResidualStage, UpConvResidualStage

class Generator(nn.Module):
    def __init__(self,
                 p_channels=3,
                 m_channels=3,
                 n_channels=1,
                 hid_channels=64,
                 out_channels=3):
        super().__init__()
        self.InputPoint=ConvReLU(p_channels, hid_channels//4, kernel_size=3, stride=1)
        self.InputMap=ConvReLU(m_channels, hid_channels//4, kernel_size=3, stride=1)
        self.InputNoise=ConvReLU(n_channels, hid_channels//2, kernel_size=3, stride=1)
        
        self.DownBlock0 = ResidualStage(hid_channels)
        
        self.DownBlock1 = ConvResidualStage(1* hid_channels, 2* hid_channels)
        self.DownBlock2 = ConvResidualStage(2* hid_channels, 4* hid_channels)
        self.DownBlock3 = ConvResidualStage(4* hid_channels, 8* hid_channels)

        self.UpBlock3 = UpConvResidualStage(8* hid_channels, 4* hid_channels)
        self.UpBlock2 = UpConvResidualStage(4* hid_channels, 2* hid_channels)
        self.UpBlock1 = UpConvResidualStage(2* hid_channels, 1* hid_channels)
        
        self.UpBlock0  =  ConvResidualStage(2* hid_channels, hid_channels//2, kernel_size=3, stride=1)
        
        self.Output=ConvTanh(hid_channels//2, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, points, maps, noises):
        p = self.InputPoint(points)
        m = self.InputMap(maps)
        n = self.InputNoise(noises)
        npm = torch.cat([n, p, m], dim=1)
        
        d0 = self.DownBlock0(npm)
        d1 = self.DownBlock1(d0)
        d2 = self.DownBlock2(d1)
        d3 = self.DownBlock3(d2)
        
        up3 = self.UpBlock3(d3)
        up2 = self.UpBlock2(up3)
        up1 = self.UpBlock1(up2)
        up0 = self.UpBlock0(torch.cat([up1, d0], dim=1))
        
        out = self.Output(up0)
        return out