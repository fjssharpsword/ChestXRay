# encoding: utf-8
import numpy as np
from os import listdir
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import os
import pickle
from collections import defaultdict
from collections import OrderedDict
from PIL import Image, ImageDraw
import PIL.ImageOps
import skimage
from skimage.io import *
from skimage.transform import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches
import matplotlib.pyplot as plt


class ROIGenerator(object):
    
    def __init__(self, TRAN_SIZE, TRAN_CROP):
        self.TRAN_SIZE = TRAN_SIZE
        self.TRAN_CROP = TRAN_CROP
        self.roi_pool = ops.RoIPool(output_size=(TRAN_CROP, TRAN_CROP), spatial_scale=1)

    # generate class activation mapping for the predicted classed
    def AnchorGen(self, feature_conv, weight_softmax, image, label): 
        # generate the class activation maps upsample to 224x224
        size_upsample = (self.TRAN_CROP, self.TRAN_CROP)
        bz, nc, h, w = feature_conv.shape
        rois, roi_labels = torch.FloatTensor(), torch.FloatTensor()
        for i in range(bz):
            idxs = np.where(label[i]==1)[0]
            for idx in idxs:
                cam = weight_softmax[idx].dot(feature_conv[i].reshape((nc,h*w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                cam_img = cv2.resize(cam_img, size_upsample)
                x_c, y_c = self.returnBox(cam_img)
                roi = self.roi_pool(image, [])

                rois = torch.cat((rois, roi), 0)
                roi_label =  np.zeros(len(label[i]))
                roi_label[idx] = 1
                roi_labels = torch.cat((roi_labels, torch.FloatTensor(roi_label)), 0)
                    
        return rois, roi_labels
    
    def returnBox(self, data): #predicted bounding boxes
        # Find local maxima
        neighborhood_size = 5 #10 #100
        threshold = .1

        data_max = filters.maximum_filter(data, neighborhood_size) #local maximum
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size) #local minimum
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        for _ in range(5):
            maxima = binary_dilation(maxima) 
        labeled, num_objects = ndimage.label(maxima)
        #slices = ndimage.find_objects(labeled)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
        #return the hot points
        x_c, y_c, data_xy = 0, 0, 0.0
        for pt in xy:
            if data[int(pt[0]), int(pt[1])] > data_xy:
                data_xy = data[int(pt[0]), int(pt[1])]
                x_c = int(pt[0])
                y_c = int(pt[1]) 
        return x_c, y_c


if __name__ == "__main__":
    #for debug   
    pass