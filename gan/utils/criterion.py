import torch
import torch.nn as nn

###############################################
############## Dice Loss ######################
###############################################
class DiceLoss(nn.Module):
    def __init__(self, alpha=10, smooth=1):
        super().__init__()
        self.alpha=alpha
        self.smooth=smooth
        
    def sigmoid(self, x):
        return 1./(1. + torch.exp(-self.alpha* x))
        
    def forward(self, fake, real):
        fake = self.sigmoid(fake)
        
        intersection = (fake * real).sum() + self.smooth
        union = fake.sum() + real.sum() + self.smooth
        dice = torch.div(2*intersection, union)
        loss = 1. - dice
        return loss
    
###############################################
############## IoU Loss ######################
###############################################
class IoUnionLoss(nn.Module):
    def __init__(self, alpha=10, smooth=1):
        super().__init__()
        self.alpha=alpha
        self.smooth=smooth
        
    def sigmoid(self, x,):
        return 1./(1. + torch.exp(-self.alpha* x))
        
    def forward(self, fake, real):
        fake = self.sigmoid(fake)
        
        intersection = (fake * real).sum() + self.smooth
        union = fake.sum() + real.sum() + self.smooth
        iou = torch.div(intersection, (union - intersection))
        loss = 1. - iou
        return loss

class PixelwiseLossMSE(nn.Module):
    def __init__(self, alpha=20):
        super().__init__()
        self.alpha=alpha

    def forward(self, fake, real):
        return self.alpha* torch.mean((fake - real)**2)
    
class PixelwiseLossL1(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha=alpha
        self.criterion=nn.L1Loss()

    def forward(self, fake, real):
        return self.alpha* self.criterion(fake, real)
    
class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.adv_criterion=nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target=torch.zeros_like(fake_pred)
        real_target=torch.ones_like(real_pred)
        fake_loss = self.adv_criterion(fake_pred, fake_target)
        real_loss = self.adv_criterion(real_pred, real_target)
        loss=(fake_loss + real_loss)/2
        return loss
    
class GeneratorLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.adv_criterion=nn.BCEWithLogitsLoss()
        self.alpha=alpha
        self.beta=beta
        
    def forward(self, fake_mpred, fake_ppred):
        fake_mtarget = torch.ones_like(fake_mpred)
        fake_ptarget = torch.ones_like(fake_ppred)
        
        map_loss = self.adv_criterion(fake_mpred, fake_mtarget)
        point_loss = self.adv_criterion(fake_ppred, fake_mpred)
        
        loss = self.alpha* map_loss + self.beta* point_loss
        return loss
    
class AdaptiveGeneratorLoss(nn.Module):
    def __init__(self, alpha=3):
        super().__init__()
        self.adv_criterion=nn.BCEWithLogitsLoss()
        self.alpha=alpha
        
    def forward(self, fake_mpred, fake_ppred, d_mloss, d_ploss):
        fake_mtarget = torch.ones_like(fake_mpred)
        fake_ptarget = torch.ones_like(fake_ppred)
        
        map_loss = self.adv_criterion(fake_mpred, fake_mtarget)
        point_loss = self.adv_criterion(fake_ppred, fake_mpred)
        
        map_coef = (self.alpha* d_mloss)/(d_ploss + self.alpha* d_mloss)
        point_coef = (d_ploss)/(d_ploss + self.alpha* d_mloss)
        
        loss = map_coef* map_loss + point_coef* point_loss
        return loss