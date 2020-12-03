# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 06/12/2020
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from roi_align import RoIAlign      # RoIAlign module

#self-defined
from ChestXRay8 import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from Utils import compute_AUCs, compute_ROCCurve
from Models.CVTEDRNet import ImgClsNet, ROIClsNet
from RPN.RPN import RegionProposalNetwork
#from Models.TripletRankingLoss import TripletRankingLoss

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='CVTEDRNet', help='CVTEDRNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = len(CLASS_NAMES)
MAX_EPOCHS = 20
BATCH_SIZE = 256 + 256
ROI_CROP = 7

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CVTEDRNet':
        IMGModel = ImgClsNet(num_classes=N_CLASSES, is_pre_trained=True)#initialize model 
        IMGModel = nn.DataParallel(IMGModel).cuda()  # make model available multi GPU cores training

        RPNModel = RegionProposalNetwork(1, 1, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16).cuda()
        roi_align = RoIAlign(ROI_CROP, ROI_CROP)
        
        ROIModel = ROIClsNet(num_classes=N_CLASSES, roi_crop = ROI_CROP)
        ROIModel = nn.DataParallel(ROIModel).cuda()
    else: 
        print('No required model')
        return #over

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    bce_criterion = nn.BCELoss() #define binary cross-entropy loss
    #tr_criterion = TripletRankingLoss() #define triplet ranking loss
    optimizer_img = optim.Adam(IMGModel.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_img = lr_scheduler.StepLR(optimizer_img, step_size = 10, gamma = 1)
    optimizer_roi = optim.Adam(ROIModel.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_roi = lr_scheduler.StepLR(optimizer_roi, step_size = 10, gamma = 1)
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    CKPT_PATH = '' #path for pre-trained model
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        IMGModel.train()  #set model to training mode
        ROIModel.train()
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):  
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                optimizer_img.zero_grad()
                #image 
                fea, img_cls = IMGModel(var_image)#forward
                loss_img = bce_criterion(img_cls, var_label) 
                loss_img.backward()#retain_graph=True
                optimizer_img.step()#update parameters    
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_img.item()) ))
                sys.stdout.flush()
                #RPN
                h, w = image.shape[2:]
                _, _, rois, roi_indices, _ = RPNModel(fea, [h, w])
                var_crops = roi_align(var_image, torch.from_numpy(rois).cuda(), torch.from_numpy(roi_indices).cuda()) #
                #ROI
                #var_crops = torch.autograd.Variable(crops).cuda()
                optimizer_roi.zero_grad()
                roi_cls = ROIModel(var_crops)
                loss_roi = bce_criterion(roi_cls, var_label[roi_indices]) 
                loss_roi.backward()#retain_graph=True
                optimizer_roi.step()#update parameters    
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_roi.item()) ))
                sys.stdout.flush()

                loss_tensor = loss_img + loss_roi
                train_loss.append(loss_tensor.item())
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        IMGModel.eval()#turn to test mode
        ROIModel.eval()
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                label = label.cuda()
                gt = torch.cat((gt, label), 0)
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                img_cls, roi_cls, roi_idxs = model(var_image)#forward
                loss_cls = bce_criterion(img_cls, var_label) 
                loss_roi = bce_criterion(roi_cls, var_label[roi_idxs]) 
                loss_tensor = loss_cls + loss_roi
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss ={}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
                pred = torch.cat((img_cls, var_output.data), 0)
                val_loss.append(loss_tensor.item())
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        print("\r Eopch: %5d validation loss = %.6f, Validataion AUROC = %.4f" % (epoch + 1, np.mean(val_loss), AUROC_avg)) 

        if AUROC_best < AUROC_avg:
            AUROC_best = AUROC_avg
            CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
            #torch.save(model.state_dict(), CKPT_PATH)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        lr_scheduler_img.step()  #about lr and gamma
        lr_scheduler_roi.step()
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
    return CKPT_PATH

def Test(CKPT_PATH = ''):
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CVTEDRNet':
        model = CVTEDRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
    else: 
        print('No required model')
        return #over

    #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded model checkpoint: "+CKPT_PATH)
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    # switch to evaluate mode
    model.eval()
    cudnn.benchmark = True
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            label = label.cuda()
            gt = torch.cat((gt, label), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            img_cls, _, _ = model(var_image)#forward
            pred = torch.cat((img_cls, var_output.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
            
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROCs[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    #Evaluating the threshold of prediction
    #thresholds = compute_ROCCurve(gt, pred, CLASS_NAMES)
    #print(thresholds)

def main():
    CKPT_PATH = Train() #for training
    #CKPT_PATH ='./Pre-trained/'+ args.model +'/best_model.pkl' #for debug
    Test(CKPT_PATH) #for test

if __name__ == '__main__':
    main()