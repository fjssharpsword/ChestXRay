# encoding: utf-8
"""
Deep product quantization of pathological regions for abnormality detection of ChestXRay in CVTE.
Author: Jason.Fang
Update time: 15/01/2021
"""
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torchvision.ops import RoIAlign
#defined by myself

# define ConvAutoencoder architecture
class PQNet(nn.Module):
    def __init__(self):
        super(PQNet, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.AvgPool2d(2, 2)
        
        #latent layer
        self.dnpool = nn.MaxPool2d(kernel_size=8, stride=8, return_indices=True)
        self.uppool = nn.MaxUnpool2d(kernel_size=8, stride=8)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

        #define common layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        #latent vector for similarity comparison
        vec, indices  = self.dnpool(x)
        x = self.uppool(vec, indices)
        #vec = vec.view(vec.size(0),-1)
        vec = vec.view(vec.size(0), vec.size(1), vec.size(2)*vec.size(3))
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = self.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = self.sigmoid(self.t_conv2(x))

        return vec, x

#A CNN Variational Autoencoder in PyTorch
#https://github.com/sksq96/pytorch-vae/blob/master/vae.py 

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = PQNet()#.to(torch.device('cuda:%d'%7))
    vec, out = model(x)
    print(vec.size())
    print(out.size())