import torch
import torch.nn as nn
import torch.nn.functional as F

##################################
######### ConvBn block ###########
##################################
    
class ConvBn(nn.Module):
    '''
    Conv2d + BatchNorm2d

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=3): kernel_size of Conv2d
        stride (default: int=1): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_bn=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        return self.conv_bn(x)
    
##################################
#######   ConvReLU block   #######
##################################

class ConvReLU(nn.Module):
    '''
    Conv2d + ReLU

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
        self.conv_relu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU())
        
    def forward(self, x):
        return self.conv_relu(x)
    
##################################
###### ConvLeakyReLU block #######
##################################

class ConvLeakyReLU(nn.Module):
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
            nn.LeakyReLU(alpha))
        
    def forward(self, x):
        return self.conv_lrelu(x)
    
##################################
#######     ConvTanh block #######
##################################

class ConvTanh(nn.Module):
    '''
    Conv2d + Tanh

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=7): kernel_size of Conv2d
        stride (default: int=1): stride of Conv2d
        padding (default: int=3): padding of Conv2d
        padding_mode (default: str=reflect): padding_mode of Conv2d
        
    '''
    def __init__(self, in_channels, out_channels,
                       kernel_size=7, stride=1,
                       padding=3, padding_mode='reflect'):
        super().__init__()
        self.conv_tanh = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, padding_mode=padding_mode),
            nn.Tanh())
        
    def forward(self, x):
        x = self.conv_tanh(x)
        return x
    
##################################
#######   ConvBnReLU block #######
##################################

class ConvBnReLU(nn.Module):
    '''
    Conv2d + BatchNorm2d + ReLU

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
        self.conv_bn_relu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        return self.conv_bn_relu(x)
    
##################################
#### ConvBnLeakyReLU block  ######
##################################

class ConvBnLeakyReLU(nn.Module):
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
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1, alpha=0.2):
        super().__init__()
        self.conv_bn_lrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha))
        
    def forward(self, x):
        return self.conv_bn_lrelu(x)
    
##################################
####### ConvBnPReLU block ########
##################################

class ConvBnPReLU(nn.Module):
    '''
    Conv2d + BatchNorm2d + PReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=4): kernel_size of Conv2d
        stride (default: int=2): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_bn_prelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU())
        
    def forward(self, x):
        return self.conv_bn_prelu(x)
    
##################################
######### Residual Block #########
##################################
    
class ResidualBlock(nn.Module):
    '''
    Residual Block (consists of 2 blocks)
        block1: Conv2d + BatchNorm2d + PReLU
        block2: Conv2d + BatchNorm2d

    Args:
        in_channels: Number of input and output channels is the same
        kernel_size (default: int=3): kernel_size of Conv2d's in block1 and block2
        stride (default: int=1): stride of Conv2d's in block1 and block2
        padding (default: int=1): padding of Conv2d's in block1 and block2
        
    '''
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnPReLU(in_channels, in_channels, kernel_size, stride, padding),
            ConvBn(in_channels, in_channels, kernel_size, stride, padding))

    def forward(self, x):
        fx = self.block(x)
        return fx + x
    
##################################
####### UpConvBnReLU block #######
##################################

class UpConvBnReLU(nn.Module):
    '''
    ConvTranspose2d + BatchNorm2d + ReLU

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
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        return self.upconv_bn_relu(x)

##################################
#######   Residual Stage   #######
##################################
    
class ResidualStage(nn.Module):
    '''
    Residual Stage (consists of 2 Residual Blocks)
        resblock0: Residual Block
        resblock1: Residual Block

    Args:
        hid_channels: Number of input and output channels (is the same) for Residual Blocks
        
    '''
    def __init__(self, hid_channels=128,):
        super().__init__()
        self.resblock0 = ResidualBlock(hid_channels, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResidualBlock(hid_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x0):
        fx0 = self.resblock0(x0)
        x1 = fx0 + x0
        fx1 = self.resblock1(x1)
        return fx1 + x1
    
##################################
#######  Conv Residual Stage #####
##################################
    
class ConvResidualStage(nn.Module):
    '''
    ConvBnReLU + Residual Stage (consists of 2 Residual Blocks)
        convblock: ConvBnReLU
        resblock0: Residual Block
        resblock0: Residual Block

    Args:
        in_channels (default: int=64): Number of input channels for ConvBnReLU
        hid_channels (default: int=128): Number of input and output channels (is the same) for Residual Blocks
        kernel_size (default: int=4): kernel_size of Conv2d in ConvBnReLU
        stride (default: int=2): stride of Conv2d in ConvBnReLU
        
    '''
    def __init__(self, in_channels=64, hid_channels=128, kernel_size=4, stride=2):
        super().__init__()
        self.convblock = ConvBnReLU(in_channels, hid_channels, kernel_size, stride)
        self.resblock0 = ResidualBlock(hid_channels, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResidualBlock(hid_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x0 = self.convblock(x)
        fx0 = self.resblock0(x0)
        x1 = fx0 + x0
        fx1 = self.resblock1(x1)
        return fx1 + x1
    
##################################
#####  UpConv Residual Stage #####
##################################

class UpConvResidualStage(nn.Module):
    '''
    UpConvBnReLU + Residual Stage (consists of 2 Residual Blocks)
        upconvblock: UpConvBnReLU
        resblock0: Residual Block
        resblock0: Residual Block

    Args:
        in_channels (default: int=64): Number of input channels for UpConvBnReLU
        hid_channels (default: int=128): Number of input and output channels (is the same) for Residual Blocks
        
    '''
    def __init__(self, in_channels=64, hid_channels=128,):
        super().__init__()
        self.upconvblock = UpConvBnReLU(in_channels, hid_channels, kernel_size=4, stride=2)
        self.resblock0 = ResidualBlock(hid_channels, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResidualBlock(hid_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x0 = self.upconvblock(x)
        fx0 = self.resblock0(x0)
        x1 = fx0 + x0
        fx1 = self.resblock1(x1)
        return fx1 + x1
    
##################################
#####     Self Attention     #####
##################################
    
class SelfAttention(nn.Module):
    '''
    SelfAttention block

    Args:
        hid_channels: Number of input channels for COnv2d's
        alpha (default: float=0): Parameter of how much attention will be added to the input
        
    '''
    def __init__(self, hid_channels, alpha=0):
        super().__init__()
        self.Q_conv = nn.Conv2d(hid_channels, hid_channels//8, kernel_size=1)
        self.K_conv = nn.Conv2d(hid_channels, hid_channels//8, kernel_size=1)
        self.V_conv = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.softmax  = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.tensor(alpha).float())
        
    def forward(self, x):
        B, C, W, H = x.size()
        
        xq = self.Q_conv(x).view(B, -1, W*H).permute(0,2,1)
        xk = self.K_conv(x).view(B, -1, W*H)
        xv = self.V_conv(x).view(B, -1, W*H)
        
        energy = torch.bmm(xq, xk)
        attention = self.softmax(energy)
        
        out = torch.bmm(xv, attention.permute(0,2,1))
        out = out.view(B, C, W, H)
        out = self.alpha* out + x
        return out
