# encoding: utf-8
"""
Attention-Guided Network for ChesstXRay 
Author: Jason.Fang
Update time: 11/16/2020
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
class ATNet(nn.Module):
    def __init__(self, num_classes, ROI_CROP=7, is_pre_trained=True):
        super(ATNet, self).__init__()
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
        
    
        """
        x = self.msa(x) * x
        x = self.dense_net_121.features(x) #output: 1024*8*8
        x = self.fc(x) #1024*7*7->14*7*7 
        x = self.sigmoid(x)
        x = x.view(x.size(0),x.size(1),x.size(2)*x.size(3)) #14*49
        tr_out = torch.mean(x, dim=1, keepdim=True).squeeze()
        bce_out = torch.mean(x, dim=2, keepdim=True).squeeze()
        return tr_out, bce_out
        """
        """
        x = self.msa(x) * x
        x = self.dense_net_121.features(x) #output: 1024*8*8
        tr_out = self.rmac(x) #1024
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        bce_out = self.dense_net_121.classifier(x)
        return tr_out, bce_out
        """
        """
        x = self.msa(x) * x
        x = self.dense_net_121.features(x) #output: 1024*8*8
        tr_out = self.rmac(x) #1024
        bce_out = self.dense_net_121.classifier(tr_out)
        return tr_out, bce_out
        """
        """
        x = self.msa(x) * x
        x = self.dense_net_121.features(x) #output: 1024*7*7
        mac_x = self.rmac(x) #1024
        x = x.view(x.size(0),-1)
        x = self.fc(x)*mac_x
        out = self.dense_net_121.classifier(x)
        return out
        """
        """
        #x = self.msa(x) * x
        x = self.dense_net_121.features(x) #output: 1024*7*7
        x = self.rmac(x) #1024
        out = self.dense_net_121.classifier(x)
        return out
        """
        """
        out = self.msa(x) * x
        out = self.dense_net_121.features(out) #output: 1024*7*7
        img_cls = self.dense_net_121.classifier(self.fc(out.view(out.size(0),-1))) #image-level classification
        #region-level classification
        _, _, rois, roi_indices, _ = self.rpn(out, [224, 224])#imgSize = [224, 224]
        crops = self.roi_align(x, torch.from_numpy(rois))
        roi_cls = self.roicls(crops.view(crops.size(0),-1))

        return img_cls, roi_cls, roi_indices
        """

#AUROC=0.8228, batchsize=512
class MultiScaleAttention(nn.Module):#spatial attention module
    def __init__(self):
        super(MultiScaleAttention, self).__init__()
        self.aggConv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.sigmoid(self.aggConv(x))

        return x  

"""
#AUROC=0.8201, batchsize=512
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
"""
 
if __name__ == "__main__":
    #for debug   
    x = torch.rand(10, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = ATNet(num_classes=5, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    img_cls, roi_cls, roi_indices = model(x)
    print(img_cls.size())