"""
DCGAN (Radford et al. 2015)

Thomas Mortier
2021
"""
import torch
import math

import torch.nn as nn
import torch.nn.functional as F


""" Generator DCGAN """
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        s=self.args.size//8
        g_l = [
            nn.ConvTranspose2d(self.args.nz, self.args.ngf*s, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.args.ngf*(s)),
            nn.ReLU(True)
        ]
        s//=2
        while s>=1:
            g_l.extend([
                nn.ConvTranspose2d(self.args.ngf*s*2, self.args.ngf*s, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.args.ngf*s),
                nn.ReLU(True)]
            )
            s//=2
        g_l.extend([
            nn.ConvTranspose2d(self.args.ngf, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        ])
        self.g = nn.Sequential(*g_l)

    def forward(self, x):
        return self.g(x)
    
""" Discriminator DCGAN """
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        d_l = [
            nn.Conv2d(3, self.args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        s=2
        while s<=self.args.size//8:
            d_l.extend([
                nn.Conv2d(self.args.ndf*(s//2), self.args.ndf*s, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.args.ndf*s),
                nn.LeakyReLU(0.2, inplace=True), 
            ])
            s*=2
        d_l.extend([
            nn.Conv2d(self.args.ndf*self.args.size//8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ])
        self.d = nn.Sequential(*d_l)

    def forward(self, x):
        return self.d(x)
