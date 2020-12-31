# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 16/12/2020
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
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
#self-defined
from CVTECXR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from Models.CVTEDRNet import CVTEDRNet

#command parameters
parser = argparse.ArgumentParser(description='For CVTE DR dataset')
parser.add_argument('--model', type=str, default='CVTEDRNet', help='CVTEDRNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
CLASS_NAMES = ['Negative', 'Positive']
N_CLASSES = len(CLASS_NAMES)
MAX_EPOCHS = 20
BATCH_SIZE = 128 + 256

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    dataloader_val = get_validation_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CVTEDRNet':
        model = CVTEDRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer , step_size = 10, gamma = 1)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    ce_criterion = nn.BCELoss() #define binary cross-entropy loss
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    acc_best = 0.50
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_output = model(var_image)
                loss_tensor = ce_criterion(var_output, var_label)
                #backward
                loss_tensor.backward() 
                optimizer.step()
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()       
                train_loss.append(loss_tensor.item()) 
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model.eval() #turn to test mode
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred= torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_output = model(var_image)
                loss_tensor = ce_criterion(var_output, var_label)
                pred = torch.cat((pred, var_output.data), 0)
                gt = torch.cat((gt, label.cuda()), 0)
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
                val_loss.append(loss_tensor.item())
        #evaluation  
        pred_np = pred[:,1].cpu().numpy()
        gt_np = gt[:,1].cpu().numpy()
        pred_np = np.where(pred_np>0.5, 1, 0)
        acc = accuracy_score(gt_np, pred_np)
        print("\r Eopch: %5d validation loss = %.6f, Validataion Accuracy = %.4f" % (epoch + 1, np.mean(val_loss), acc)) 

        if acc_best < acc:
            acc_best = acc
            CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
            #torch.save(model.state_dict(), CKPT_PATH)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CVTEDRNet':
        model = CVTEDRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded Image model checkpoint: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    # switch to evaluate mode
    model.eval() #turn to test mode
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            var_output = model(var_image)
            pred = torch.cat((pred, var_output.data), 0)
            gt = torch.cat((gt, label.cuda()), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #for evaluation   
    for thr in [0.5, 0.4, 0.3, 0.2, 0.1]: 
        pred_np = pred[:,1].cpu().numpy()
        gt_np = gt[:,1].cpu().numpy()
        pred_np = np.where(pred_np>thr, 1, 0)
        acc = accuracy_score(gt_np, pred_np)
        tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
        sen = tp /(tp+fn)
        spe = tn /(tn+fp)
        print("\r Threshold = %.4f, Test Accuracy = %.4f" % (thr, acc)) 
        print('\r Threshold = {:.4f}: the Sensitivity = {:.4f} and the specificity = {:.4f}'.format(thr, sen, spe))

def main():
    #Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()