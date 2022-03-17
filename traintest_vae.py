"""
Code for training and testing of VAE model.

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
from mlmodels.vae import VAE
from torchsummary import summary

def tensor_to_img(X, figname=""):
    X = X.permute([0,2,3,1]).cpu().detach().numpy()
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    ax.imshow((X[0,:,:,:]*255).astype(np.uint8))
    plt.savefig(figname+".png")

# delete old png files 
for file in os.listdir('.'):
    if file.endswith('.png') and '_vae_' in file:
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

# model
model = VAE(c_in, hidden_dim, latent_dim, gamma, isotropic)

print("Number of total parameters for model = {0}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
if use_cuda:
    model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#  start training
for epoch in range(max_epochs):
    # Training
    train_loss = 0
    for X, _ in training_dataloader:
        # also draw noise samples from N(0,1)
        E = torch.randn(X.shape[0],latent_dim).to(device)
        # gpu transfer
        X = X.to(device)
        # calculate forward pass with loss
        loss, p = model(X, E)
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print("Epoch {0} loss = {1}".format(epoch+1, train_loss/len(training_dataloader)))
    tensor_to_img(p, 'reconstruction_vae_{0}'.format(epoch))
    Xhat = model.sample(torch.randn(params["batch_size"],latent_dim).to(device))
    tensor_to_img(Xhat, 'sample_vae_{0}'.format(epoch))
