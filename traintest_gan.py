"""
Code for training and testing of GAN model.

Thomas Mortier
2021
"""

import torch
import os
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from torch import nn
from mlmodels.data import AnimalDataset
from torchvision import datasets, transforms
from mlmodels.gan import GAND, GANG
from torchsummary import summary

def tensor_to_img(X, figname=""):
    X = X.permute([0,2,3,1]).cpu().detach().numpy()
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    ax.imshow((X[0,:,:,:]*255).astype(np.uint8))
    plt.savefig(figname+".png")

# delete old png files 
for file in os.listdir('.'):
    if file.endswith('.png') and '_gan_' in file:
        os.remove(file)

# cuda init
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# params
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 2}
learn_rate = 0.001
max_epochs = 1000
img_size = (64,64)
c_in = 3
hidden_dim = [32, 64, 128]
latent_dim = 5012
gamma = 0.01
isotropic = True

# data
t = transforms.Compose([transforms.ToTensor()])
data = AnimalDataset("/home/data/tfmortier/Github/mlmodels/data/Animals", img_size, t)
training_dataloader = torch.utils.data.DataLoader(data, **params)

# models
generator = GANG(c_in, hidden_dim, latent_dim)
discriminator = GANG(c_in, hidden_dim)
print("Number of total parameters for generator = {0}".format(sum(p.numel() for p in generator.parameters() if p.requires_grad)))
print("Number of total parameters for discriminator = {0}".format(sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))
if use_cuda:
    generator = generator.to(device)
    discriminator = discriminator.to(device)

# optimizers and criterion
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learn_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learn_rate)

#  start training
for epoch in range(max_epochs):
    # Training
    generator_loss = 0
    discriminator_loss = 0
    for X, _ in training_dataloader:
        # first train the discriminator
        optimizer_d.zero_grad()        
        # 1.1) D on real images
        labels = torch.full((X.size(0),), 1).to(device)
        output = discriminator(labels)
        loss_D_r = criterion(output, labels)
        loss_D_r.backward()
        optimizer_d.step()
        # 1.2) D on fake images
        labels = torch.fill_(0)
        X_fake = torch.randn(X.size(0), latent_dim).to(device)
        output = discriminator(X_fake)
        loss_D_f = criterion(output, labels)
        loss_D_f.backward() # important: gradients are accumulated (summed) with previous gradients
        loss_D = loss_D_r + loss_D_f
        discriminator_loss += loss_D.item()
        optimizer_d.step()

        # secondly train the generator
        optimizer_g.zero_grad()
        labels = torch.fill_(1)
        loss_G = criterion(output, labels)
        loss_G.backward()
        generator_loss += loss_G.item()
        optimizer_g.step()

    print("Epoch {0} generator loss = {1}        discriminator loss = {2}".format(epoch+1, generator_loss/len(training_dataloader), discriminator_loss/(len(training_dataloader)*2))
    Xhat = model.sample(torch.randn(params["batch_size"],latent_dim).to(device))
    tensor_to_img(Xhat, 'sample_gan_{0}'.format(epoch))
