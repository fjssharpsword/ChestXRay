# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 08/12/2020
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
#from Models.RMAC import RMAC
#sys.path.append('RPN')
from RPN.RPN import RegionProposalNetwork

#construct model
class CVTEDRNet(nn.Module):
    def __init__(self, num_classes, ROI_CROP=7, is_pre_trained=True):
        super(CVTEDRNet, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        self.msa = MultiScaleAttention()
        #self.rmac = RMAC(level_n=3)
        #self.fc = nn.Conv2d(num_fc_kernels, num_classes, kernel_size=3, padding=1, bias=False)
        #self.sigmoid = nn.Sigmoid()
        self.fc  = nn.Linear(1024*7*7, num_fc_kernels)
        self.rpn = RegionProposalNetwork(#conv feature 1024*7*7
            1024, 1024,
            ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            feat_stride=16
        )
        # RoIAlign layer with crop sizes:
        self.roi_align = RoIAlign(output_size=(ROI_CROP, ROI_CROP), spatial_scale=1.0, sampling_ratio=-1)
        self.roicls = nn.Sequential(nn.Linear(3*ROI_CROP*ROI_CROP, num_classes), nn.Sigmoid())
        
    def forward(self, x):
        #x: N*C*W*H
    
        x = self.msa(x) * x
        out = self.dense_net_121(x) 
        return out
        
    
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
    x = torch.rand(10, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = CVTEDRNet(num_classes=5, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    out = model(x)
    print(out.size())