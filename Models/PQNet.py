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

#construct model
class PQNet(nn.Module):
    def __init__(self, num_classes, is_pre_trained=True):
        super(PQNet, self).__init__()
        #DensetNet121
        self.dense_net_121 = torchvision.models.densenet121(pretrained=is_pre_trained)
       
    def forward(self, x):
        #x: N*C*W*H
        x = self.dense_net_121.features(x) #1024*7*7, regions of 1024 with size of 7*7
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        return x


if __name__ == "__main__":
    #for debug   
    x = torch.rand(2, 3, 224, 224)#.to(torch.device('cuda:%d'%7))
    model = PQNet(num_classes=2, is_pre_trained=True)#.to(torch.device('cuda:%d'%7))
    out = model(x)
    print(out.size())