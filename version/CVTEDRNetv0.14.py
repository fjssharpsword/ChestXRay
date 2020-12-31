# encoding: utf-8
"""
Attention-Guided Network for ChestXRay 
Author: Jason.Fang
Update time: 16/12/2020
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

#construct model
class CVTEDRNet(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(CVTEDRNet, self).__init__()

        #Spatial-Attention
        self.sa = SpatialAttention()

        #DensetNet121
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
        num_fc_kernels = self.dense_net_121.classifier.in_features #1024
        self.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Sigmoid())

        #ResNet50
        #self.resnet_50 = torchvision.models.resnet50(pretrained=is_pre_trained)
        #res_fc_features = self.resnet_50.fc.in_features
        #self.resnet_50.classifier = nn.Sequential(nn.Linear(res_fc_features, num_classes), nn.Sigmoid())
        #GoogLeNet
        #self.googlenet = torchvision.models.googlenet(pretrained=is_pre_trained)
        #goog_fc_kernels = self.googlenet.fc.in_features
        #self.googlenet.classifier = nn.Sequential(nn.Linear(goog_fc_kernels, num_classes), nn.Sigmoid())
        #unified
        #self.classifier = nn.Sequential(nn.Linear(1000, num_classes), nn.Sigmoid())
       
    def forward(self, x):
        #x: N*C*W*H
        """ 
        #densenet
        x = self.dense_net_121(x) 
        return x
        """

        """
        #densenet + sa + fully-connected layer
        x = self.sa(x) * x
        x = self.dense_net_121(x) 
        return x
        """
        #densenet + sa + fully-convolutional layer
        x = self.sa(x) * x
        x = self.dense_net_121.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1) #for 224*224
        x = self.dense_net_121.classifier(x) 
        return x
        """
        #Ensemble learning, overfitting
        x = self.sa(x) * x
        out_dense = self.dense_net_121(x)
        out_res = self.resnet_50(x)
        out_goog = self.googlenet(x)

        out_fusion = out_dense + out_res + out_goog #weight
        out_fusion = F.relu(out_fusion, inplace=True)
        out = self.classifier(out_fusion)
        return out
        """
          
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
    out = model(x)
    print(out.size())