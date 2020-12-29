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
from skimage.measure import label
from sklearn.metrics import accuracy_score, confusion_matrix
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
MAX_EPOCHS = 10
BATCH_SIZE = 128 + 32

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
    ce_criterion = nn.CrossEntropyLoss() #nn.BCELoss() #define binary cross-entropy loss
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
                loss_tensor = ce_criterion(var_output, var_label.squeeze())
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
        gt = torch.LongTensor().cuda()
        pred= torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_output = model(var_image)
                loss_tensor = ce_criterion(var_output, var_label.squeeze())
                pred = torch.cat((pred, var_output.data), 0)
                gt = torch.cat((gt, label.cuda()), 0)
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
                val_loss.append(loss_tensor.item())
        #evaluation     
        pred = F.log_softmax(pred.cpu().numpy(), dim=1) 
        pred = pred.max(1,keepdim=True)[1]  
        acc = accuracy_score(gt.cpu().numpy(), pred)
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
    gt = torch.LongTensor().cuda()
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
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    pred_np = F.log_softmax(pred_np, dim=1) 
    pred_np = pred_np.max(1,keepdim=True)[1]  
    acc = accuracy_score(gt_np, pred_np)
    tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
    sen = tp /(tp+fn)
    spe = tn /(tn+fp)
    print("\r Test Accuracy = %.4f" % (acc)) 
    print('The Sensitivity is {:.4f} and the specificity is {:.4f}'.format(sen, spe))

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()