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
        # conv layer (depth from 3 --> 128), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # conv layer (depth from 128 --> 256), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # conv layer (depth from 256 --> 512), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # conv layer (depth from 512 --> 1024), 3x3 kernels
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.AvgPool2d(2, 2)
        
        #latent layer
        self.dnpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(16, 3, 2, stride=2)

        #common layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.msa = MultiScaleAttention()

    def forward(self, x):
        ##Attention Layer##
        x = self.msa(x)*x

        ## encode ##
        # add hidden layers with relu activation function
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x))) 

        #latent vector for similarity comparison
        vec, indices  = self.dnpool(x)
        x = self.uppool(vec, indices)
        #vec = vec.view(vec.size(0),-1)
        vec = vec.view(vec.size(0), vec.size(1), vec.size(2)*vec.size(3))
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        x = self.relu(self.t_conv3(x))
        x = self.relu(self.t_conv4(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = self.sigmoid(x)

        return vec, x

class MultiScaleAttention(nn.Module):#multi-scal attention module
    def __init__(self):
        super(MultiScaleAttention, self).__init__()
        
        self.scaleConv1 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False)
        self.scaleConv2 = nn.Conv2d(3, 3, kernel_size=9, padding=4, bias=False)
        
        self.aggConv = nn.Conv2d(6, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        out_avg = torch.mean(x, dim=1, keepdim=True)
        
        out1 = self.scaleConv1(x)
        out_max1, _ = torch.max(out1, dim=1, keepdim=True)
        out_avg1 = torch.mean(out1, dim=1, keepdim=True)
        
        out2 = self.scaleConv2(x)
        out_max2, _ = torch.max(out2, dim=1, keepdim=True)
        out_avg2 = torch.mean(out2, dim=1, keepdim=True)

        x = torch.cat([out_max, out_avg, out_max1, out_avg1, out_max2, out_avg2], dim=1)
        x = self.sigmoid(self.aggConv(x))

        return x

#A CNN Variational Autoencoder in PyTorch
#https://github.com/sksq96/pytorch-vae/blob/master/vae.py 

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = PQNet()#.to(torch.device('cuda:%d'%7))
    vec, out = model(x)
    print(vec.size())
    print(out.size())