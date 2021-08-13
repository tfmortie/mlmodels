"""
Normalizing flow implemenation based on Glow (Kingma et al. 2018)
Thomas Mortier
2021
"""
import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import scipy as sp
import numpy as np

from torch.distributions import Normal

class ActnormLayer(nn.Module):
    """ Represents an actnorm layer.
    """
    def __init__(self, c_in, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, z):
        with torch.no_grad():
            flatten = z.permute(1, 0, 2, 3).contiguous().view(z.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            _, _, height, width = z.shape

            if self.initialized.item() == 0:
                self.initialize(z)
                self.initialized.fill_(1)

            log_abs = torch.log(torch.abs(self.scale))
            ldj += height * width * torch.sum(log_abs)
            return self.scale * (z + self.loc), ldj

        else:
            log_abs = torch.log(torch.abs(self.scale))
            _, _, height, width = z.shape
            ldj -= height * width * torch.sum(log_abs)
            return z / self.scale - self.loc, ldj

class OBOConvLayer(nn.Module):
    """ Represents an invertible 1x1 convolutional layer.
    """
    def __init__(self, dim, device):
        super().__init__()
        w = torch.randn(dim, dim).to(device)
        w = torch.qr(w)[0]  
        self.w = nn.Parameter(w).to(device)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            B,C,H,W = z.shape
            logdet = torch.slogdet(self.w)[-1] * H * W
            ldj += logdet
            return F.conv2d(z, self.w.view(C,C,1,1)), ldj
        else:
            B,C,H,W = z.shape
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(self.w)[-1] * H * W
            ldj += logdet
            return F.conv2d(z, w_inv.view(C,C,1,1)), ldj

class ZeroConvLayer(nn.Module):
    """ Represents the last convolutional layer of the coupling network.
    """
    def __init__(self, c_in, c_out, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(c_in, c_out, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def forward(self, z):
        out = F.pad(z, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out

class AffineCouplingLayer(nn.Module):
    """ Represents an affine coupling layer. 

        c_in : number of input channels for inputs
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.couplingnetwork = nn.Sequential(
            nn.Conv2d(self.c_in//2, self.c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c_out, self.c_out, 1),
            nn.ReLU(inplace=True),
            ZeroConvLayer(self.c_out, self.c_in)
        )
        self.couplingnetwork[0].weight.data.normal_(0, 0.05)
        self.couplingnetwork[0].bias.data.zero_()
        self.couplingnetwork[2].weight.data.normal_(0, 0.05)
        self.couplingnetwork[2].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # first split z 
        z_a, z_b = z.chunk(2,dim=1)
        # obtain parameters for coupling transform
        th_a_s, th_a_t = self.couplingnetwork(z_a).chunk(2, dim=1)
        if not reverse:
            # apply coupling transform
            th_a_s = torch.sigmoid(th_a_s+2)
            z_b = z_b*th_a_s + th_a_t
            ldj += th_a_s.log().sum(dim=[1,2,3])
            return torch.cat([z_a, z_b], axis=1), ldj
        else:
            th_a_s = torch.sigmoid(th_a_s+2)
            z_b = (z_b - th_a_t)/th_a_s
            ldj += -th_a_s.log().sum(dim=[1,2,3])
            return torch.cat([z_a, z_b], axis=1), ldj

class SqueezeLayer(nn.Module):
    """ Represents the squeeze operation for the multi-scale flow.
    """
    def forward(self, z, ldj, reverse=False):
        B, C, H, W = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H//2, 2, W//2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4*C, H//2, W//2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C//4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C//4, H*2, W*2)
        return z, ldj

class SplitLayer(nn.Module):
    """ Represents the split operation for the multi-scale flow.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device 
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
            ldj += self.prior.log_prob(z_split).sum(dim=[1,2,3])
        else:
            z_split = self.prior.sample(sample_shape=z.shape).to(self.device)
            z = torch.cat([z, z_split], dim=1)
            ldj += self.prior.log_prob(z_split).sum(dim=[1,2,3])
        return z, ldj

class NF(torch.nn.Module):
    """ Represents the main NF class.
    """
    def __init__(self, K, L, c_in, n_hidden, device):
        super(NF, self).__init__()
        self.flows = nn.ModuleList([])
        for i in range(L-1):
            self.flows.append(SqueezeLayer())
            for _ in range(K):
                self.flows.append(ActnormLayer(c_in*(2**i)))
                self.flows.append(OBOConvLayer(c_in*(2**i), device))
                self.flows.append(AffineCouplingLayer(c_in*(2**i), n_hidden))
            self.flows.append(SplitLayer(device))
        self.flows.append(SqueezeLayer())
        for _ in range(K):
            self.flows.append(ActnormLayer(c_in*(2**(L-1))))
            self.flows.append(OBOConvLayer(c_in*(2**(L-1)), device))
            self.flows.append(AffineCouplingLayer(c_in*(2**(L-1)), n_hidden))
        # represents or prior distribution on the latent space
        self.prior = Normal(0,1)
        # device
        self.device = device

    def forward(self, x):
        return self.log_likelihood(x)

    def encode(self, x):
        z, ldj = x, torch.zeros((x.shape[0])).to(self.device)
        for i,f in enumerate(self.flows):
            z, ldj = f(z, ldj)

        return z, ldj

    def decode(self, z):
        ldj = torch.zeros((z.shape[0])).to(self.device)
        for f in self.flows[::-1]:
            z, ldj = f(z, ldj, reverse=True)

        return z, ldj

    def log_likelihood(self, x):
        z, ldj = self.encode(x)
        pz = self.prior.log_prob(z).sum(dim=[1,2,3])
        nll = -(pz+ldj)
        nll = (nll*math.log2(math.exp(1)))/x[0].numel()

        return nll

    def sample(self, z):
        return self.decode(z)  
