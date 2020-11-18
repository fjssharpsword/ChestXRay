# encoding: utf-8
import re
import sys
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
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
from PIL import Image, ImageDraw
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import matplotlib.patches as patches
import PIL.ImageOps

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


def VisualizationLesion(CKPT_PATH):
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
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded model checkpoint: "+CKPT_PATH)
    print('******************** load model succeed!********************')

    print('******* begin bounding box visulization!*********')   
    model.eval()# switch to evaluate mode
    cls_weights = list(model.parameters())
    weight_softmax = np.squeeze(cls_weights[-2].data.cpu().numpy()) 
    with torch.autograd.no_grad():
        for batch_idx, (image_name, gtbox, image, label) in enumerate(dataloader_bbox):
            var_image = torch.autograd.Variable(image).cuda()
            var_feature = model.dense_net_121.features(var_image) #get feature maps
            logit = model(var_image)#forword
            h_x = F.softmax(logit, dim=1).data.squeeze()#softmax
            probs, idx = h_x.sort(0, True) #probabilities of classe
            cam_img = returnCAM(var_feature.cpu().data.numpy(), weight_softmax, idx[0].item())
            predBoxes = genPredBoxes(cam_img, gtbox[0])
            #plot
            heat_map = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
            raw_img = ReadRawImage(image_name[0])
            #output_img = cv2.addWeighted(raw_img, 0.7, heat_map, 0.3, 0)
            output_img = OverlapImages(gtbox[0].numpy(), predBoxes, heat_map, raw_img) #crop heatmap and merge
            fig, ax = plt.subplots(1)# Create figure and axes
            ax.imshow(output_img)
            x,y,w,h = gtbox[0][0], gtbox[0][1], gtbox[0][2], gtbox[0][3]
            rect = patches.Rectangle((x, y), w, h,linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
            ax.add_patch(rect)# Add the patch to the Axes
            for i in range(len(predBoxes)):
                x_p,y_p,w_p,h_p = predBoxes[i][0], predBoxes[i][1], predBoxes[i][2], predBoxes[i][3]
                IoU_score, _ = compute_IoUs_and_Dices([x,y,w,h],[x_p,y_p,w_p,h_p])
                rect = patches.Rectangle((x_p, y_p), w_p, h_p,linewidth=2, edgecolor='b', facecolor='none')# Create a Rectangle patch
                ax.add_patch(rect)# Add the patch to the Axes
                ax.text(x_p, y_p, '{:.4f}'.format(IoU_score))
            ax.axis('off')

            gt_idx = np.where(label[0]==1)[0][0]
            gt_label = np.array(CLASS_NAMES)[gt_idx]
            gt_idx = np.where(idx.cpu().numpy()==gt_idx)[0][0]
            img_file = 'gt-{}-{:.4f}_pred-{}-{:.4f}'.format(gt_label, probs[gt_idx].cpu().item(), CLASS_NAMES[idx[0].item()], probs[0].cpu().item())
            fig.savefig('./Imgs/'+img_file+'.jpg')
            sys.stdout.write('\r Visualization process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

def OverlapImages(gtbox, predBoxes, heat_map, raw_img):
    """
    heat_map = Image.fromarray(heat_map)#PIL.Image
    heat_map = PIL.ImageOps.invert(heat_map).convert('RGBA')#add alpha channel
    raw_img = Image.fromarray(raw_img)#PIL.Image
    raw_img = PIL.ImageOps.invert(raw_img).convert('RGBA')
    #transparency
    datas = heat_map.getdata()
    newData = list()
    for item in datas:
        if item[0] ==255 and item[1] ==255 and item[2] ==255: #white
            newData.append(( 255, 255, 255, 0))#Transparency
        elif item[0] ==0 and item[1] ==0 and item[2] ==0: #black
            newData.append(( 0, 0, 255, 100))    #opaque  to blue
        else:#gray
            #newData.append(( item[0],item[1],item[2], 100))
            newData.append(( 0 , 255, 0, 100)) #green
    heat_map.putdata(newData)
    overlay = Image.alpha_composite(raw_img, heat_map)
    overlay = overlay.convert('RGB')
    """
    """
    heat_map = Image.fromarray(heat_map)#PIL.Image
    raw_img = Image.fromarray(raw_img)#PIL.Image
    mask=Image.new('L', heat_map.size, color=255)  
    draw=ImageDraw.Draw(mask) 
    draw.rectangle(transparent_area, fill=0) 
    heat_map.putalpha(mask) 
    overlay = Image.alpha_composite(raw_img, heat_map)
    """
    heat_map = Image.fromarray(heat_map)#PIL.Image
    roi_area = (int(gtbox[0]), int(gtbox[1]), int(gtbox[0]+gtbox[2]), int(gtbox[1]+gtbox[3]))
    cropped_roi = heat_map.crop(roi_area)
    #mask_img = Image.new('RGB', heat_map.size, color='white')
    mask_img = Image.new('RGBA', heat_map.size, color=0) #transparency
    mask_img.paste(cropped_roi, roi_area)
    for i in range(len(predBoxes)):
        roi_area = (int(predBoxes[i][0]), int(predBoxes[i][1]), int(predBoxes[i][0]+predBoxes[i][2]), int(predBoxes[i][1]+predBoxes[i][3]))
        cropped_roi = heat_map.crop(roi_area)
        mask_img.paste(cropped_roi, roi_area)
    output_img = cv2.addWeighted(raw_img, 0.5, np.asarray(mask_img.convert('RGB')), 0.5, 0)
    return output_img

def ReadRawImage(image_name):
    raw_img = Image.open(image_name).convert('RGB')
    width, height = raw_img.size   # Get dimensions
    x_scale = 256/width
    y_scale = 256/height
    raw_img = raw_img.resize((256, 256),Image.ANTIALIAS)
    width, height = raw_img.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    crop_del = (256-224)/2
    raw_img = raw_img.crop((left, top, right, bottom)) 
    raw_img= np.array(raw_img)
    return raw_img

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
    predBoxes = []
    for pt in xy:
        if data[int(pt[0]), int(pt[1])] > np.max(data)*.9:
            x_p = int(max(pt[0], 0.))
            y_p = int(max(pt[1], 0.))
            w_p = int(min(x_p + w, 224)) - x_p
            h_p = int(min(y_p + h, 224)) - y_p
            predBoxes.append([x_p,y_p,w_p,h_p])
    return predBoxes

def main():
    CKPT_PATH ='./Pre-trained/'+ args.model +'/best_model.pkl' #for debug 
    VisualizationLesion(CKPT_PATH) #for location

if __name__ == '__main__':
    main()