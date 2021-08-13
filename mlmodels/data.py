import os
import cv2
import torch
import pandas as pd
from torchvision.io import read_image
from skimage import io, transform, color

import matplotlib.pyplot as plt

class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        # process data
        dirs_tp = [x[0] for x in os.walk(self.img_dir) if "Head" in x[0]]
        # init data
        self.img_labels = []
        self.img_paths = []
        # go over each dir (which represents a label)
        for d in dirs_tp:
            # extract label
            lbl = d.split("/")[-1].split("Head")[0].lower()
            # run over each file in dir
            for filepath in os.listdir(d):
                # only extract rgb images
                image = io.imread(os.path.join(d,filepath))
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        self.img_labels.append(lbl)
                        self.img_paths.append(os.path.join(d,filepath))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = io.imread(self.img_paths[idx])
        image = transform.resize(image, self.img_size)
        if image.shape != (*self.img_size, 3):
            print(image.shape)
        #skimage.color.grey2rgb(image, alpha=None)
        image, label = self.transform(image), self.img_labels[idx]

        return image.float(), label
