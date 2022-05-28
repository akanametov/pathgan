import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import inception_v3
from .functional import kl_divergence, covariance, frechet_distance 


class KLDivergence(nn.Module):
    """Kullback–Leibler divergence."""
    def __init__(self,):
        super().__init__()
        
    def forward(self, px, py):
        return kl_divergence(px, py)


class InceptionScore(nn.Module):
    """Inception Score metrics

    Parameters
    ----------
    device: str, (default="cpu")
        torch.device
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        encoder = self.__loadModel__()
        self.encoder = self.__freeze__(encoder).to(device)
        self.downsize = lambda x: F.interpolate(x, size=(299,299), mode="bilinear")
        self.upsize = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)
    
    @staticmethod
    def __loadModel__():
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()
        return model.eval() 
    
    @staticmethod
    def __freeze__(model):
        for p in model.parameters():
            p.requires_grad = False
        return model
        
    def forward(self, x):
        assert x.dim() == 4
        
        dim = x.size(2)
        if  dim < 299:
            x = self.upsize(x)
        elif dim > 299:
            x = self.downsize(x)

        feats = self.encoder(x)
        px = F.softmax(feats, dim=1)
        py = px.mean(0, keepdims=True)
        kl_div = kl_divergence(px, py)
        sum_kl = kl_div.sum(axis=1)
        avg_kl = sum_kl.mean()
        iscore = avg_kl.exp()
        return iscore


class FrechetInceptionDistance(nn.Module):
    """Frechet Inception Distance.

    Parameters
    ----------
    device: str, (default="cpu")
        torch.device
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        encoder = self.__loadModel__()
        self.encoder = self.__freeze__(encoder).to(device)
        self.downsize = lambda x: F.interpolate(x, size=(299,299), mode="bilinear")
        self.upsize = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)
    
    @staticmethod
    def __loadModel__():
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()
        return model.eval() 
    
    @staticmethod
    def __freeze__(model):
        for p in model.parameters():
            p.requires_grad = False
        return model
        
    def forward(self, x, y):
        assert (x.dim() == 4) and (y.dim() == 4)
        
        dim = x.size(2)
        if  dim < 299:
            x = self.upsize(x)
        elif dim > 299:
            x = self.downsize(x)
            
        dim = y.size(2)
        if  dim < 299:
            y = self.upsize(y)
        elif dim > 299:
            y = self.downsize(y)

        fake_feats = self.encoder(x)
        real_feats = self.encoder(y)
        
        mu_fake = fake_feats.mean(0)
        sigma_fake = covariance(fake_feats)
        
        mu_real = real_feats.mean(0)
        sigma_real = covariance(real_feats)
        return frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
