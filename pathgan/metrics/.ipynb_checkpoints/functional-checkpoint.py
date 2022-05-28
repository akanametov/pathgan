import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import inception_v3


def covariance(x):
    d = x.shape[-1]
    mean = torch.mean(x, dim=-1).unsqueeze(-1)
    x = x - mean
    return 1/(d-1) * x @ x.transpose(-1, -2)


def correlation(x, eps=1e-08):
    d = x.shape[-1]
    mean = torch.mean(x, dim=-1).unsqueeze(-1)
    std = torch.std(x, dim=-1).unsqueeze(-1)
    x = (x - mean) / (std + eps)
    return 1/(d-1) * x @ x.transpose(-1, -2)


def sqrtm(A):
    P, V = torch.symeig(A, eigenvectors=True)
    dtype_cond = {torch.float32: 1e-6, torch.float64: 1e-10}
    cond = dtype_cond[A.dtype]
    cutoff = (abs(P) > cond * torch.max(abs(P)))

    D = torch.sqrt(P[cutoff])
    S = torch.diag(D)
    V = V[:, cutoff]
    B = V @ S @ V.t()
    return B


def kl_divergence(px, py):
    eps = 1e-8
    kl_div = px*(torch.log(px + eps) - torch.log(py + eps))
    return kl_div


def frechet_distance(muA, sigmaA, muB, sigmaB):
    ssum = (muA - muB) @ (muA - muB)
    fd = ssum + (sigmaA.trace() + sigmaB.trace() - 2*(sqrtm(sigmaA @ sigmaB)).trace())
    return fd
