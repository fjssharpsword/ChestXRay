# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 06/12/2020
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
from torchvision.ops import RoIAlign, RoIPool

#construct model
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

class ImgClsNet(nn.Module):
    def __init__(self, num_classes, roi_crop=7, is_pre_trained=True):
        super(ImgClsNet, self).__init__()
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())
        self.sa = SpatialAttention()
    
    def forward(self, x):
        #x: N*C*W*H
        fea = self.sa(x) #1*W*H
        img_cls = self.dense_net_121(fea*x)
        
        return fea, img_cls

class ROIClsNet(nn.Module):#ROI module
    def __init__(self, num_classes, roi_crop):
        super(ROIClsNet, self).__init__()
        #self.roi_align = RoIAlign(output_size=(roi_crop, roi_crop), spatial_scale=1.0, sampling_ratio=-1)
        #self.roi_pool = RoIPool(output_size=(roi_crop, roi_crop), spatial_scale=1)
        self.ROIClassifier = nn.Sequential(nn.Linear(3*roi_crop*roi_crop, num_classes), nn.Sigmoid())

    def forward(self, x):
        #x: regional features 
        roi_cls = self.ROIClassifier(x.view(x.size(0),-1))
        return roi_cls

if __name__ == "__main__":
    #for debug   
    x = torch.rand(10, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = ImgClsNet(num_classes=5, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    fea, img_cls = model(x)
    print(fea.size())