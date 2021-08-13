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
from mlmodels.nf import NF
from mlmodels.nfnew import Glow

def tensor_to_img(X, figname=""):
    X = X.permute([0,2,3,1]).cpu().detach().numpy()
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    ax.imshow((X[0,:,:,:]*255).astype(np.uint8))
    plt.savefig(figname+".png")

# delete old png files 
for file in os.listdir('.'):
    if file.endswith('.png'):
        os.remove(file)

# cuda init
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# params
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 2}
learn_rate = 0.0001
max_epochs = 200
img_size = (64,64)
debug = True
FS=256
K=16
L=3
complexmodel = False

# data
t = transforms.Compose([transforms.ToTensor()])
data = AnimalDataset("/home/data/tfmortier/Github/mlmodels/data/Animals", img_size, t)
training_dataloader = torch.utils.data.DataLoader(data, **params)

# model
if complexmodel:
    model = Glow(FS,K,L,input_dims=(3,*img_size),gaussianize=True,OBO=True)
else:
    model = NF(K,L,12,FS,device)

print("Number of total parameters for model = {0}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
if use_cuda:
    model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

#  start training
for epoch in range(max_epochs):
    # Training
    train_loss = 0
    for X, _ in training_dataloader:
        # gpu transfer
        X = X.to(device)

        if complexmodel:
            loss = - model.log_prob(X, bits_per_pixel=True).mean(0)
        else:
            loss = model(X).mean(0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()
        train_loss += loss.item()

    print("Epoch {0} loss = {1}".format(epoch+1, train_loss/len(training_dataloader)))

    if complexmodel:
        Xhat = model.inverse(batch_size=32)[0]
    else:
        Xhat, _ = model.sample(torch.randn(params["batch_size"],12*(2**(L-1)),img_size[0]//(2**L),img_size[0]//(2**L)).to(device))
    
    tensor_to_img(Xhat, 'sample_nf_{0}'.format(epoch))
