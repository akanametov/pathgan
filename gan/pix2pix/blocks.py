import torch
import torch.nn as nn
    
##################################
####### ConvLeakyReLU block ######
##################################

class ConvLReLU(nn.Module):
    '''
    Conv2d + LeakyReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=4): kernel_size of Conv2d
        stride (default: int=2): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        alpha (default: float=0.2): alpha of LeakyReLU
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1, alpha=0.2):
        super().__init__()
        self.conv_lrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(alpha, inplace=True))
        
    def forward(self, x):
        return self.conv_lrelu(x)
    
##################################
######### ConvTanh block #########
##################################

class ConvTanh(nn.Module):
    '''
    Conv2d + Tanh

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=1): kernel_size of Conv2d
        stride (default: int=1): stride of Conv2d
        padding (default: int=0): padding of Conv2d
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv_th=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Tanh())
        
    def forward(self, x):
        return self.conv_th(x)
    
##################################
##### ConvBnLeakyReLU block ######
##################################

class ConvBnLReLU(nn.Module):
    '''
    Conv2d + BatchNorm2d + LeakyReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=4): kernel_size of Conv2d
        stride (default: int=2): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        alpha (default: float=0.2): alpha of LeakyReLU
        
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1, alpha=0.2):
        super().__init__()
        self.conv_bn_lrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(alpha, inplace=True))
        
    def forward(self, x):
        return self.conv_bn_lrelu(x)
    
##################################
####### UpConvBnReLU block #######
##################################

class UpConvBnReLU(nn.Module):
    '''
    ConvTranspose2d + InstanceNorm2d (replaced BatchNorm2d) + ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=4): kernel_size of Conv2d
        stride (default: int=2): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.upconv_bn_relu=nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        return self.upconv_bn_relu(x)
