"""
Variational autoencoder implementation (Kingma et al. 2013)

Thomas Mortier
2021
"""

import torch
import math

import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """ represents the main vae class.
    """
    def __init__(self, c_in, hidden_dim, latent_dim, gamma, isotropic, device):
        super(VAE, self).__init__()
        # store information
        self.c_in = c_in 
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        self.gamma = gamma # controls the influence of the reconstruction loss
        self.isotropic = isotropic # defines whether the latent space is an isotropic Guassian or full-covariance
        
        # register encoder
        self.encoder = []
        input_channels = self.c_in
        for d in self.hidden_dim:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(input_channels, d, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(),
            ))
            input_channels = d
        self.encoder = nn.Sequential(*self.encoder)
        # register variational parameters for recognition model/encoder  
        self.mu_f = nn.Linear(self.hidden_dim[-1]*16, self.latent_dim)
        self.logsigma_f = nn.Linear(self.hidden_dim[-1]*16, self.latent_dim)
        if not self.isotropic:
            self.Lt_f = nn.Linear(self.hidden_dim[-1]*16, self.latent_dim*self.latent_dim)
            # register mask 
            self.register_buffer('Lm',torch.tril(torch.ones(self.latent_dim,self.latent_dim),diagonal=-1))

        # register decoder
        self.decoder_pp = nn.Linear(latent_dim, self.hidden_dim[-1]*16)
        self.decoder = []
        for i in range(1,len(self.hidden_dim)):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim[-i],self.hidden_dim[-i-1],kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.BatchNorm2d(self.hidden_dim[-i-1]),
                nn.LeakyReLU()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(self.hidden_dim[0],self.hidden_dim[0],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dim[0],out_channels=self.c_in,kernel_size=3,padding=1),
            nn.Sigmoid()))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x, e):
        # encode
        h = self.encoder(x)
        h = h.view(h.shape[0],-1)
        mu, logsigma = self.mu_f(h), self.logsigma_f(h)
        if not self.isotropic:
            Lt = self.Lt_f(h)
            Lt = Lt.view(h.shape[0],self.Lm.shape[0], self.Lm.shape[1])
            # calculate L
            L = self.Lm*Lt+torch.diag_embed(logsigma.exp(), offset=0, dim1=-2, dim2=-1)
            # calcalate L @ e
            Le = torch.einsum('bij,bi->bi',L, e)
            z = Le + mu
        else:
            z = (logsigma.exp()*e)+mu

        # prior to decoding make sure that our hidden representation is extended to correct format for last convolutional layer in encoder
        z = self.decoder_pp(z)
        z = z.view(z.shape[0],self.hidden_dim[-1],4,4)
        
        # decode and calculate loss
        p = self.decoder(z)
        kl_loss = (-0.5*(1+logsigma-mu**2-torch.exp(logsigma)).sum(dim=1)).mean(dim=0)/(self.hidden_dim[-1]*4*4)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(p,x)
        loss = recon_loss+kl_loss*self.gamma

        return loss, p

    def sample(self, e):
        # decode 
        with torch.no_grad():
            p = self.decoder_pp(e)
            p = p.view(p.shape[0],self.hidden_dim[-1],4,4)
            p = self.decoder(p)
        return p
