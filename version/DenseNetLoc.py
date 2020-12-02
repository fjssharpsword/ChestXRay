# encoding: utf-8
import re
import sys
import os
import cv2
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from skimage.measure import label
from PIL import Image
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

from ChestXRay8 import get_bbox_dataloader
from Utils import compute_IoUs_and_Dices, GradCAM, GradCamPlusPlus
from Models.DenseNet import DenseNet121

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='DenseNet', help='DenseNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 1

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define
    def forward(self, x):
        return self.module(x)

def WeaklyLocation(CKPT_PATH):
    print('********************load data********************')
    dataloader_bbox = get_bbox_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'DenseNet':
        model = DenseNet121(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model
    else: 
        print('No required model')
        return #over

    #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
    #model = WrappedModel(model)
    #for name, layer in model.named_modules():
        #if isinstance(layer, nn.ReLU):
        #print(name, layer)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded model checkpoint: "+CKPT_PATH)
    print('******************** load model succeed!********************')

    print('******* begin bounding box location!*********')   
    model.eval()# switch to evaluate mode
    cls_weights = list(model.parameters())
    weight_softmax = np.squeeze(cls_weights[-2].data.cpu().numpy()) 
    classes = {0: 'Atelectasis', 1: 'Cardiomegaly', 2: 'Effusion', 3: 'Infiltration', 4:'Mass', 5:'Nodule', 6:'Pneumonia',\
                7:'Pneumothorax',8:'Consolidation',9:'Edema',10:'Emphysema',11:'Fibrosis',12:'Pleural_Thickening',13:'Hernia'}
    IoUs = []
    Dices = []
    #grad_cam = GradCAM(net=model, layer_name='dense_net_121.features.denseblock4.denselayer16.conv2')
    #grad_cam_plus_plus = GradCamPlusPlus(net=model, layer_name='dense_net_121.features.denseblock4.denselayer16.conv2')
    with torch.autograd.no_grad():#enable_grad() for GramCAM
        for batch_idx, (_, gtbox, image, label) in enumerate(dataloader_bbox):
            var_image = torch.autograd.Variable(image).cuda()
            var_feature = model.dense_net_121.features(var_image) #get feature maps
            #var_feature = grad_cam_plus_plus(var_image)
            #grad_cam_plus_plus.remove_handlers()
            logit = model(var_image)#forword
            h_x = F.softmax(logit, dim=1).data.squeeze()#softmax
            probs, idx = h_x.sort(0, True) #probabilities of classe
            cls_prob_line = '{:.4f} -> {}'.format(probs[0], classes[idx[0].item()])
            cam_img = returnCAM(var_feature.cpu().data.numpy(), weight_softmax, idx[0].item())
            IoU_Score,  Dice_Score= genPredBoxes(cam_img, gtbox[0]) 
            # genPredBoxes(var_feature, gtbox[0]) for GramCAM
            IoUs.append(IoU_Score.item())
            Dices.append(Dice_Score.item())
            sys.stdout.write('\r Location process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    print('IoUs = {:.4f} and Dices = {:.4f}'.format(np.mean(np.array(IoUs)), np.mean(np.array(Dices))))  

# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 224x224
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape

    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc,h*w)))
    #cam = weight_softmax[class_idx]*(feature_conv.reshape((nc,h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)
    return cam_img

def genPredBoxes(data, gtbox): #predicted bounding boxes
    w, h = gtbox[2], gtbox[3]
    # Find local maxima
    neighborhood_size = 100
    threshold = .1

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    #for _ in range(5):
    #    maxima = binary_dilation(maxima) 
    labeled, num_objects = ndimage.label(maxima)
    #slices = ndimage.find_objects(labeled)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
    IoU_Score = 0.0
    Dice_Score = 0.0 
    for pt in xy:
        if data[int(pt[0]), int(pt[1])] > np.max(data)*.9:
            x_p = int(max(pt[0], 0.))
            y_p = int(max(pt[1], 0.))
            w_p = int(min(x_p + w, 224)) - x_p
            h_p = int(min(y_p + h, 224)) - y_p
            IoU_T, Dice_T = compute_IoUs_and_Dices(gtbox, [x_p,y_p,w_p,h_p])
            IoU_Score += IoU_T
            Dice_Score += Dice_T
    return IoU_Score, Dice_Score

def main():
    CKPT_PATH ='./Pre-trained/'+ args.model +'/best_model.pkl' #for debug 
    WeaklyLocation(CKPT_PATH) #for location

if __name__ == '__main__':
    main()