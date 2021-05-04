import torch
from torch import nn
from .blocks import ConvTanh, ConvLReLU, ConvBnLReLU, UpConvBnReLU

class Generator(nn.Module):
    '''
    ROI Generator

    Args:
        map_channels (default: int=3): Number of Map input channels 
        point_channels (default: int=3): Number of Point input channels 
        hid_channels (default: int=32): Number of hidden channels
        out_channels (default: int=3): Number of output (ROI) channels
    '''
    def __init__(self,
                 map_channels=3, 
                 point_channels=3,
                 hid_channels=32,
                 out_channels=3):
        super().__init__()
        self.InputMap=ConvLReLU(map_channels, hid_channels//2, kernel_size=4)
        self.InputPoint=ConvLReLU(point_channels, hid_channels//2, kernel_size=4)
        
        self.DownBlock1=ConvBnLReLU(hid_channels, 2*hid_channels, kernel_size=4)
        self.DownBlock2=ConvBnLReLU(2*hid_channels, 4*hid_channels, kernel_size=4)
        self.DownBlock3=ConvBnLReLU(4*hid_channels, 8*hid_channels, kernel_size=4)
        self.DownBlock4=ConvBnLReLU(8*hid_channels, 8*hid_channels, kernel_size=4)
        
        self.UpBlock5=nn.Sequential(
                UpConvBnReLU(8*hid_channels, 8*hid_channels, kernel_size=4),
                nn.Dropout2d(0.5))
        self.UpBlock4=nn.Sequential(
                UpConvBnReLU(16*hid_channels, 4*hid_channels, kernel_size=4),
                nn.Dropout2d(0.5))
        self.UpBlock3=UpConvBnReLU(8*hid_channels, 2*hid_channels, kernel_size=4)
        self.UpBlock2=UpConvBnReLU(4*hid_channels, hid_channels, kernel_size=4)
        self.UpBlock1=UpConvBnReLU(2*hid_channels, 2*out_channels, kernel_size=4)
        
        self.Output=ConvTanh(4*out_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, maps, points):
        x0 = torch.cat([maps, points], 1)
        m = self.InputMap(maps)
        p = self.InputPoint(points)
        x1 = torch.cat([m, p], 1)
        x2 = self.DownBlock1(x1)
        x3 = self.DownBlock2(x2)
        x4 = self.DownBlock3(x3)
        x5 = self.DownBlock4(x4)
        
        y5 = self.UpBlock5(x5)
        y4 = self.UpBlock4(torch.cat([y5, x4], 1))
        y3 = self.UpBlock3(torch.cat([y4, x3], 1))
        y2 = self.UpBlock2(torch.cat([y3, x2], 1))
        y1 = self.UpBlock1(torch.cat([y2, x1], 1))
        
        y0 = self.Output(torch.cat([y1, x0], 1))
        return y0



class Discriminator(nn.Module):
    '''
    Discriminator

    Args:
        in_channels (default: int=9): Number of Map + Point + Fake/Real ROI channels 
        hid_channels (default: int=32): Number of hidden channels
        out_channels (default: int=1): Number of output channels
    '''
    def __init__(self,
                 in_channels=9,
                 hid_channels=32,
                 out_channels=1):
        super().__init__()
        self.Input=nn.Identity()
        
        self.DownBlock1=ConvLReLU(in_channels, hid_channels, kernel_size=4)
        self.DownBlock2=ConvBnLReLU(hid_channels, 2*hid_channels, kernel_size=4)
        self.DownBlock3=ConvBnLReLU(2*hid_channels, 4*hid_channels, kernel_size=4)
        self.DownBlock4=nn.Sequential(
            nn.ZeroPad2d(1),
            ConvBnLReLU(4*hid_channels, 8*hid_channels, kernel_size=4, stride=1),
            nn.ZeroPad2d(1))
        
        self.Output=nn.Conv2d(8*hid_channels, out_channels, kernel_size=4, stride=1)
        
    def forward(self, x):
        y0 = self.Input(x)
        
        y1 = self.DownBlock1(y0)
        y2 = self.DownBlock2(y1) 
        y3 = self.DownBlock3(y2) 
        y4 = self.DownBlock4(y3) 
        
        y5 = self.Output(y4)
        return y5
    
class SDiscriminator(nn.Module):
    '''
    Discriminator (with separate blocks for Map/Point/ROI

    Args:
        map_channels (default: int=3): Number of Map input channels 
        point_channels (default: int=3): Number of Point input channels 
        roi_channels (default: int=3): Number of ROI input channels 
        hid_channels (default: int=32): Number of hidden channels
        out_channels (default: int=1): Number of output channels
    '''
    def __init__(self,
                 map_channels=3, 
                 point_channels=3,
                 roi_channels=3,
                 hid_channels=32,
                 out_channels=1):
        super().__init__()
        self.InputMap=ConvLReLU(map_channels, hid_channels//4, kernel_size=4)
        self.InputPoint=ConvLReLU(point_channels, hid_channels//4, kernel_size=4)
        self.InputROI=ConvLReLU(roi_channels, hid_channels//2, kernel_size=4)
        
        self.DownBlock1=ConvBnLReLU(hid_channels, 2*hid_channels, kernel_size=4)
        self.DownBlock2=ConvBnLReLU(2*hid_channels, 4*hid_channels, kernel_size=4)
        self.DownBlock3=nn.Sequential(
            nn.ZeroPad2d(1),
            ConvBnLReLU(4*hid_channels, 8*hid_channels, kernel_size=4, stride=1),
            nn.ZeroPad2d(1))
        
        self.Output=nn.Conv2d(8*hid_channels, out_channels, kernel_size=4, stride=1)
        
    def forward(self, maps, points, rois):
        m = self.InputMap(maps)
        p = self.InputPoint(points)
        r = self.InputROI(rois)
        y0 = torch.cat([m, p, r], dim=1)
        
        y1 = self.DownBlock1(y0)
        y2 = self.DownBlock2(y1) 
        y3 = self.DownBlock3(y2)  
        
        y4 = self.Output(y3)
        return y4
