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
from Models.RMAC import RMAC

#construct model
class PQNet(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(PQNet, self).__init__()
        #Spatial-Attention
        self.sa = SpatialAttention()
        #DensetNet121
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        #RMAC
        self.rmac = RMAC(level_n=3)
       
    def forward(self, x):
        #x: N*C*W*H
        #densenet + sa + fully-convolutional layer
        x = self.sa(x) * x
        x = self.dense_net_121.features(x)
        mac = self.rmac(x)
        out = F.relu(x, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(out.size(0), -1) #for 224*224
        out = self.dense_net_121.classifier(out) 
        return x, mac, out

          
class SpatialAttention(nn.Module):#spatial attention module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.aggConv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.aggConv(x))

        return x  

if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = CVTEDRNet(num_classes=2, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    fea, out = model(x)
    print(fea.size())
    print(out.size())