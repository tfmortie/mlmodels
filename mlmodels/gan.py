"""
Generative adversarial networks (Goodfellow et al. 2014)

Thomas Mortier
2021
"""

import torch
import math

import torch.nn as nn
import torch.nn.functional as F


class GANG(nn.Module):
    """ Represents the GAN generator class.
    """
    def __init__(self, c_in, hidden_dim, latent_dim):
        super(GANG, self).__init__()
        self.c_in = c_in
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # register generator
        self.generator_pp = nn.Linear(latent_dim, self.hidden_dim[-1]*64)
        self.generator = []
        for i in range(1,len(self.hidden_dim)):
            self.generator.append(nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim[-i],self.hidden_dim[-i-1],kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.BatchNorm2d(self.hidden_dim[-i-1]),
                nn.LeakyReLU()))
        self.generator.append(nn.Sequential(nn.ConvTranspose2d(self.hidden_dim[0],self.hidden_dim[0],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dim[0],out_channels=self.c_in,kernel_size=3,padding=1),
            nn.Sigmoid()))
        self.generator = nn.Sequential(*self.generator)

    def forward(self, e):
        p = self.generator_pp(e)
        p = p.view(e.size(0),self.hidden_dim[-1],8,8)
        p = self.generator(p)
        return p

class GAND(nn.Module):
    """ Represents the GAN discriminator class.
    """
    def __init__(self, c_in, hidden_dim):
        super(GAND, self).__init__()
        self.c_in = c_in
        self.hidden_dim = hidden_dim
        
        # register discriminator
        self.discriminator = []
        input_channels = self.c_in
        for d in self.hidden_dim:
            self.discriminator.append(nn.Sequential(
                nn.Conv2d(input_channels, d, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(),
            ))
            input_channels = d
        self.discriminator_pp = nn.Sequential(
            nn.Linear(self.hidden_dim[-1]*64, 1),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(*self.discriminator)

    def forward(self, x):
        p = self.discriminator(x)
        p = p.view(x.size(0),-1)
        p = self.discriminator_pp(p)

        return p
