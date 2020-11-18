# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 11/11/2020
"""
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

#construct model
class ATNet(nn.Module):
    def __init__(self, num_classes, is_pre_trained):
        super(ATNet, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        self.ca = ChannelAttention(3)#1024
        self.sa = SpatialAttention()
        self.fc = nn.Linear(1024*7*7, num_fc_kernels)
        self.downsample = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)#nn.MaxPool2d
        self.bn = nn.BatchNorm2d(3) #channel=3
        
    def forward(self, x):
        #Scale1: 3*256*256
        out = self.sa(x) * x
        #fea1 = self.dense_net_121.features(xs1)#1024*7*7
        out = self.dense_net_121(out) 
        out = torch.unsqueeze(out, 0)
        #Scale2: 3*128*128
        x = self.downsample(x)
        x = self.bn(x)
        xs = self.sa(x) * x
        xs = self.dense_net_121(xs) 
        xs = torch.unsqueeze(xs, 0)
        out = torch.cat((out, xs), 0)
        #Scale3: 3*64*64
        x = self.downsample(x)
        x = self.bn(x)
        xs = self.sa(x) * x
        xs = self.dense_net_121(xs) 
        xs = torch.unsqueeze(xs, 0)
        out = torch.cat((out, xs), 0)

        out = torch.mean(out, dim=0, keepdim=True) #aggregation torch.max or according to gradient
        out = torch.squeeze(out, 0)
        return out


#https://github.com/Andy-zhujunwen/pytorch-CBAM-Segmentation-experiment-/blob/master/model/resnet_cbam.py
class SpatialAttention(nn.Module):#spatial attention layer
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_channels=in_planes, out_channels=in_planes // 2, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels=in_planes // 2, out_channels=in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
