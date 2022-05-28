import torch
from torch import nn, Tensor


class DICELoss(nn.Module):
    """DICELoss.

    Parameters
    ----------
    alpha: float, (default=10.0)
        Coefficient in exp of sigmoid function.  
    smooth: float, (default=1.0)
        To prevent zero in nominator.
    """
    def __init__(self, alpha: float = 10.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        
    def sigmoid(self, x: Tensor):
        return 1.0/(1.0 + torch.exp(-self.alpha * x))
        
    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        fake = self.sigmoid(fake)

        intersection = (fake * real).sum() + self.smooth
        union = fake.sum() + real.sum() + self.smooth
        dice = torch.div(2.0*intersection, union)
        loss = 1.0 - dice
        return loss


class IoULoss(nn.Module):
    """Intersection over Union loss.

    Parameters
    ----------
    alpha: float, (default=10.0)
        Coefficient in exp of sigmoid function.  
    smooth: float, (default=1.0)
        To prevent zero in nominator.
    """
    def __init__(self, alpha: float = 10.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        
    def sigmoid(self, x: Tensor):
        return 1.0/(1.0 + torch.exp(-self.alpha * x))
        
    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        fake = self.sigmoid(fake)
        
        intersection = (fake * real).sum() + self.smooth
        union = fake.sum() + real.sum() + self.smooth
        iou = torch.div(intersection, (union - intersection))
        loss = 1. - iou
        return loss


class PixelwiseLossMSE(nn.Module):
    """MSE loss function

    Parameters
    ----------
    alpha: float, (default=20.0)
        Coefficient by which loss will be multiplied
    """
    def __init__(self, alpha: float = 20.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        return self.alpha* torch.mean((fake - real)**2)


class PixelwiseLossL1(nn.Module):
    """L1 loss.

    Parameters
    ----------
    alpha: float, (default=1.0)
        Coefficient by which loss will be multiplied
    """
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha=alpha
        self.criterion=nn.L1Loss()

    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        return self.alpha * self.criterion(fake, real)


class GeneratorLoss(nn.Module):
    """Generator (BCE) loss function

    Parameters
    ----------
    alpha: float, (default=100)
        Coefficient by which loss will be multiplied
    """
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha=alpha
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()
        
    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha* self.l1(fake, real)
        return loss


class SAGeneratorLoss(nn.Module):
    """Generator (BCE) loss function

    Parameters
    ----------
    alpha: float, (default=1.0)
        Coefficient by which map loss will be multiplied
    beta: float, (default=1.0)
        Coefficient by which point loss will be multiplied
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, fake_mpred: Tensor, fake_ppred: Tensor) -> Tensor:
        fake_mtarget = torch.ones_like(fake_mpred)
        fake_ptarget = torch.ones_like(fake_ppred)
        map_loss = self.adv_criterion(fake_mpred, fake_mtarget)
        point_loss = self.adv_criterion(fake_ppred, fake_mpred)
        loss = self.alpha * map_loss + self.beta * point_loss
        return loss


class AdaptiveSAGeneratorLoss(nn.Module):
    """Adaptive Generator (BCE) loss.

    Parameters
    ----------
    alpha: float, (default=3.0)
        Coefficient by which loss will be multiplied
    """
    def __init__(self, alpha: float = 3.0):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        
    def forward(self, fake_mpred: Tensor, fake_ppred: Tensor, d_mloss: Tensor, d_ploss: Tensor):
        fake_mtarget = torch.ones_like(fake_mpred)
        fake_ptarget = torch.ones_like(fake_ppred)
        
        map_loss = self.adv_criterion(fake_mpred, fake_mtarget)
        point_loss = self.adv_criterion(fake_ppred, fake_mpred)
        
        map_coef = (self.alpha * d_mloss)/(d_ploss + self.alpha * d_mloss)
        point_coef = (d_ploss)/(d_ploss + self.alpha * d_mloss)
        
        loss = map_coef * map_loss + point_coef * point_loss
        return loss


class DiscriminatorLoss(nn.Module):
    """Discriminator (BCE) loss."""
    def __init__(self,):
        super().__init__()
        self.adv_criterion=nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred: Tensor, real_pred: Tensor) -> Tensor:
        fake_target=torch.zeros_like(fake_pred)
        real_target=torch.ones_like(real_pred)
        fake_loss = self.adv_criterion(fake_pred, fake_target)
        real_loss = self.adv_criterion(real_pred, real_target)
        loss=(fake_loss + real_loss)/2
        return loss
