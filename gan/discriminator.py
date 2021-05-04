import torch
from torch import nn
from .modules import ConvLeakyReLU, ConvBnLeakyReLU
from .modules import SelfAttention

class MapDiscriminator(nn.Module):
    '''
    The Map Discriminator

    Args:
        map_channels (default: int=3): Number of Map input channels 
        roi_channels (default: int=3): Number of ROI input channels
        hid_channels (default: int=64): Number of hidden channels
        out_channels (default: int=1): Number of output channels
    '''
    def __init__(self,
                 map_channels=3,
                 roi_channels=3,
                 hid_channels=64,
                 out_channels=1):
        super().__init__()
        self.InputMap=ConvBnLeakyReLU(map_channels, hid_channels//2, kernel_size=4, stride=2, padding=1)
        self.InputRegion=ConvLeakyReLU(roi_channels, hid_channels//2, kernel_size=4, stride=2, padding=1)
        
        self.MapAttention=SelfAttention(hid_channels//2)
        self.RegionAttention=SelfAttention(hid_channels//2)
        
        self.Block0=ConvBnLeakyReLU(1* hid_channels, 2* hid_channels, kernel_size=4, stride=2, padding=1)
        self.Block1=ConvBnLeakyReLU(2* hid_channels, 4* hid_channels, kernel_size=4, stride=2, padding=1)
        self.Attention=SelfAttention(4* hid_channels)
        self.Block2=ConvBnLeakyReLU(4* hid_channels, 8* hid_channels, kernel_size=4, stride=2, padding=1)
        
        self.Output=nn.Conv2d(8* hid_channels, out_channels, kernel_size=4, stride=1)
        
    def forward(self, maps, regions):
        m = self.InputMap(maps)
        r = self.InputRegion(regions)
        ma = self.MapAttention(m)
        ra = self.RegionAttention(r)
        fm = m + ma
        fr = r + ra
        mr = torch.cat([fm, fr], dim=1)
        
        x = self.Block0(mr)
        x = self.Block1(x)
        xa = self.Attention(x)
        fx = x + xa
        x = self.Block2(fx)
        out = self.Output(x)
        return out
    
class PointDiscriminator(nn.Module):
    '''
    The Point Discriminator

    Args:
        point_channels (default: int=3): Number of Point input channels 
        roi_channels (default: int=3): Number of ROI input channels
        hid_channels (default: int=64): Number of hidden channels
        out_channels (default: int=1): Number of output channels
    '''
    def __init__(self,
                 point_channels=3,
                 roi_channels=3,
                 hid_channels=64,
                 out_channels=1):
        super().__init__()
        self.InputPoint=ConvBnLeakyReLU(point_channels, 3*hid_channels//4, kernel_size=4, stride=2, padding=1)
        self.InputRegion=ConvLeakyReLU(roi_channels,  1*hid_channels//4, kernel_size=4, stride=2, padding=1)
        
        self.PointAttention=SelfAttention(3*hid_channels//4)
        self.RegionAttention=SelfAttention(1*hid_channels//4)
        
        self.Block0=ConvBnLeakyReLU(1* hid_channels, 2* hid_channels, kernel_size=4, stride=2, padding=1)
        self.Block1=ConvBnLeakyReLU(2* hid_channels, 4* hid_channels, kernel_size=4, stride=2, padding=1)
        self.Attention=SelfAttention(4* hid_channels)
        self.Block2=ConvBnLeakyReLU(4* hid_channels, 8* hid_channels, kernel_size=4, stride=2, padding=1)
        
        self.Output=nn.Conv2d(8* hid_channels, out_channels, kernel_size=4, stride=1)
        
    def forward(self, points, regions):
        p = self.InputPoint(points)
        r = self.InputRegion(regions)
        pa = self.PointAttention(p)
        ra = self.RegionAttention(r)
        fp = p + pa
        fr = r + ra
        pr = torch.cat([fp, fr], dim=1)
        
        x = self.Block0(pr)
        x = self.Block1(x)
        xa = self.Attention(x)
        fx = x + xa
        x = self.Block2(fx)
        out = self.Output(x)
        return out
