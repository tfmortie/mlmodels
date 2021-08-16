"""
Variational autoencoder implementation (Kingma et al. 2013)

Thomas Mortier
2021
"""

import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal

class VAE(torch.nn.Module):
    """ Represents the main VAE class.
    """
    def __init__(self, c_in, hidden_dim, latent_dim, device):
        super(NF, self).__init__()
        # store information
        self.c_in = c_in 
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # register encoder
        self.encoder = []
        input_channels = self.c_in
        for d in self.hidden_dim
            self.encoder.append(nn.Sequential(
                nn.Conv2d(input_channels, d, kernel_size=3, stride=2, padding=1),
                nn.Batchnorm2d(d),
                nn.LeakyReLU(),
            )
            input_channels = d
        self.encoder = nn.Sequential(*self.encoder)
        # register variational parameters for recognition model/encoder  
        self.mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.logsigma = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.Lt = nn.Linear(self.hidden_dim[-1], self.latent_dim*self.latent_dim)
        # register mask 
        self.register_buffer('Lm',torch.tril(torch.ones(self.latent_dim,self.latent_dim),diagonal=-1))

        # register decoder
        self.decoder = [nn.Linear(latent_dim, self.hidden_dim[-1])]
        for i in range(1,len(self.hidden_dim)):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-i],hidden_dims[-i-1],kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.BatchNorm2d(hidden_dims[-i-1]),
                nn.LeakyReLU())
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(self.hidden_dim[0],self.hidden_dim[0],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dim[0],out_channels=self.c_in,kernel_size=3,padding= 1),
            nn.Sigmoid())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        print("TODO")

    def sample(self, z):
        print("TODO")
