"""
Code for training and testing of DCGAN model.

Thomas Mortier
2021
"""
import torch
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torchvision import transforms
from mlmodels.dcgan import Generator, Discriminator

def traindcgan(args):
    print(f'{args=}')
    # create device (default GPU), generator and discriminator
    device = torch.device("cuda")
    G = Generator(args).to(device)
    # enable multi-gpu
    G = nn.DataParallel(G, list(range(2)))
    D = Discriminator(args).to(device)
    # enable multi-gpu
    D = nn.DataParallel(D, list(range(2)))
    g_total_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print(f'{g_total_params=}')
    d_total_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f'{d_total_params=}')
    # create dataloader
    data_transform = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=args.data, transform=data_transform)
    dataloader = DataLoader(dataset=dataset,
                          batch_size=args.batchsize,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True) 
    # optimizers and criterion
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(G.parameters(), betas=(args.beta, 0.999), lr=args.learnrateg)
    optimizer_d = torch.optim.Adam(D.parameters(), betas=(args.beta, 0.999), lr=args.learnrated)
    start_time = time.time()
    #  start training
    D = D.train()
    G = G.train()
    for epoch in range(args.epochs):
        # Training
        for i, (X, _) in enumerate(dataloader):
            # we need to maximize log(D(x)) + log(1 - D(G(z)))
            # first train the discriminator
            D.zero_grad()
            # 1.1) D on real images 
            labels = torch.full((X.size(0),), 1, dtype=torch.float).to(device)
            output = D(X.to(device))
            loss_D_r = criterion(output.view(-1), labels) 
            loss_D_r.backward()
            # 1.2) D on fake images
            labels = labels.fill_(0)
            Z = torch.randn(args.batchsize, args.nz, 1, 1).to(device)
            output = D(G(Z))
            loss_D_f = criterion(output.view(-1), labels) 
            loss_D_f.backward()
            loss_D = loss_D_r + loss_D_f
            optimizer_d.step()
            # secondly train the generator
            optimizer_g.zero_grad()
            G.zero_grad()
            labels = labels.fill_(1)
            Z = torch.randn(args.batchsize, args.nz, 1, 1).to(device)
            output = D(G(Z))
            loss_G = criterion(output.view(-1), labels)
            loss_G.backward()
            optimizer_g.step()
            if ((i+1) % args.nit) == 0:
                print("Epoch {0} G loss = {1}       D loss = {2} in {3}s".format(epoch+1, loss_G.item(), loss_D.item(), time.time()-start_time))
                start_time = time.time()
    print("Training done!")
    model_path = "./"+args.modelout+"_"+str(args.size)+"_"+str(args.nz)+".pt"
    print("Saving generator to {0}...".format(model_path))
    torch.save(G, model_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Code for DCGAN.")
    # model
    parser.add_argument("--nz", type=int, default=100, help="Number of channels of latent input.")
    parser.add_argument("--ngf", type=int, default=64, help="Size feature maps of generator.")
    parser.add_argument("--ndf", type=int, default=64, help="Size feature maps of discriminator.")
    # data
    parser.add_argument("--size", type=int, default=64, help="Size of generated images (must be equal to power of 2).")
    parser.add_argument("--data", type=str, default="/home/data/tfmortier/Research/Datasets/ART/images/images", help="Path to folder with images.")
    parser.add_argument("--modelout", type=str, default="gen")
    # training
    parser.add_argument("--nit", type=int, default=10, help="Defines number of iterations before printing results.")
    parser.add_argument("--learnrateg", type=float, default=0.0002, help="Learning rate for generator training.")
    parser.add_argument("--learnrated", type=float, default=0.0002, help="Learning rate for discriminator training.")
    parser.add_argument("--batchsize", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for Adam optimizer.")
    args = parser.parse_args()
    traindcgan(args)
