"""
Normalizing flow model 
Thomas Mortier
2021
"""

import torch

class NF(torch.nn.Module):
    def __init__(self,x):
        super(NF, self).__init__()
        self.x = x

    def forward(self, x):
        return None
