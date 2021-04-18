import torch
import torch.nn as nn
import torch.nn.functional as F

##################################
######### ConvBn block ###########
##################################
    
class ConvBn(nn.Module):
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