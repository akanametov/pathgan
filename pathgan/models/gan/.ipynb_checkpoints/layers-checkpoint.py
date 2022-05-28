"""Layers."""

import torch
from torch import nn, Tensor


class Conv(nn.Module):
    """Conv.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    kernel_size: int, (default=4)
        Kernel size of Conv2d.
    stride: int, (default=2)
        Stride of Conv2d.
    padding: int, (default=1)
        Padding of Conv2d.
    normalization: str, (default=None)
        Normalization function: ["batch", "instance"].
    activation: str, (default=None)
        Activation function: ["relu", "lrelu", "tanh"].
    alpha: float, (default=0.2)
        Alpha of LeakyReLU.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        normalization: str = None,
        activation: str = None,
        alpha: float = 0.2,
    ):
        super().__init__()
        # convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # normalization function
        if normalization is not None:
            if normalization == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif normalization == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise ValueError(f"There is no `{normalization}` normalization function.")
        else:
            self.norm = None
        # activation function
        if activation is not None:
            if activation == "relu":
                self.act = nn.ReLU(inplace=True)
            elif activation == "lrelu":
                self.act = nn.LeakyReLU(alpha, inplace=True)
            elif activation == "prelu":
                self.act = nn.PReLU()
            elif activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError(f"There is no `{activation}` activation function.")
        else:
            self.act = None
        
    def forward(self, x: Tensor) -> Tensor:
        fx = self.conv(x)
        if self.norm is not None:
            fx = self.norm(fx)
        if self.act is not None:
            fx = self.act(fx)
        return fx


class UpConv(nn.Module):
    """UpConv.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    kernel_size: int, (default=4)
        Kernel size of Conv2d.
    stride: int, (default=2)
        Stride of Conv2d.
    padding: int, (default=1)
        Padding of Conv2d.
    normalization: str, (default=None)
        Normalization function: ["batch", "instance"].
    activation: str, (default=None)
        Activation function: ["relu", "lrelu", "tanh"].
    alpha: float, (default=0.2)
        Alpha of LeakyReLU.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        normalization: str = None,
        activation: str = None,
        dropout: float = None,
        alpha: float = 0.2,
    ):
        super().__init__()
        # convolutional layer
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        # normalization function
        if normalization is not None:
            if normalization == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif normalization == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise ValueError(f"There is no `{normalization}` normalization function.")
        else:
            self.norm = None
        # activation function
        if activation is not None:
            if activation == "relu":
                self.act = nn.ReLU(inplace=True)
            elif activation == "lrelu":
                self.act = nn.LeakyReLU(alpha, inplace=True)
            elif activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError(f"There is no `{activation}` activation function.")
        else:
            self.act = None
        # dropout
        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        fx = self.upconv(x)
        if self.norm is not None:
            fx = self.norm(fx)
        if self.act is not None:
            fx = self.act(fx)
        if self.dropout is not None:
            fx = self.dropout(fx)
        return fx


class SelfAttention(nn.Module):
    """SelfAttention.

    Parameters
    ----------
    hid_channels: int
        Number of input channels for Conv2d.
    alpha: float, (default=0.0)
        Parameter of how much attention will be added to the input.
    """
    def __init__(self, hid_channels: int, alpha: float = 0.0):
        super().__init__()
        self.q_conv = nn.Conv2d(hid_channels, hid_channels//8, kernel_size=1)
        self.k_conv = nn.Conv2d(hid_channels, hid_channels//8, kernel_size=1)
        self.v_conv = nn.Conv2d(hid_channels, hid_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.tensor(alpha).float())
        
    def forward(self, x):
        B, C, W, H = x.size()

        xq = self.q_conv(x).view(B, -1, W*H).permute(0,2,1)
        xk = self.k_conv(x).view(B, -1, W*H)
        xv = self.v_conv(x).view(B, -1, W*H)

        energy = torch.bmm(xq, xk)
        attention = self.softmax(energy)

        out = torch.bmm(xv, attention.permute(0,2,1))
        out = out.view(B, C, W, H)
        out = self.alpha* out + x
        return out


class ResBlock(nn.Module):
    """Residual Block.

    Parameters
    ----------
    in_channels: int
        Number of input and output channels is the same.
    kernel_size: int, (default=3)
        Kernel size of Conv2d's.
    stride: int, (default=1)
        Stride of Conv2d's.
    padding: int, (default=1)
        Padding of Conv2d's.
    """
    def __init__(
        self,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv1 = Conv(planes, planes, kernel_size, stride, padding, normalization="instance", activation="prelu")
        self.conv2 = Conv(planes, planes, kernel_size, stride, padding, normalization="instance")

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.conv2(fx)
        fx = fx + x
        return fx


class ResStage(nn.Module):
    """Residual Stage.

    Parameters
    ----------
    hid_channels: int
        Number of input and output channels for Residual Blocks.
    """
    def __init__(self, hid_channels: int = 128, in_channels: int = None, kernel_size=4, stride=2, upscale: bool = True):
        super().__init__()
        self.in_channels = in_channels
        if in_channels is not None:
            if upscale:
                self.conv = UpConv(in_channels, hid_channels, kernel_size, stride, normalization="instance", activation="relu")
            else:
                self.conv = Conv(in_channels, hid_channels, kernel_size, stride, normalization="instance", activation="relu")
        self.res_block1 = ResBlock(hid_channels, kernel_size=3, stride=1, padding=1)
        self.res_block2 = ResBlock(hid_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_channels is not None:
            x = self.conv(x)
        fx1 = self.res_block1(x)
        fx1 = fx1 + x
        fx2 = self.res_block2(fx1)
        fx2 = fx2 + fx1
        return fx2
