# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 15/01/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
#self-defined
from CVTECXR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from Models.PQNet import PQNet

#command parameters
parser = argparse.ArgumentParser(description='For CVTE DR dataset')
parser.add_argument('--model', type=str, default='PQNet', help='PQNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3, 4, 5"
CLASS_NAMES = ['Negative', 'Positive']
N_CLASSES = len(CLASS_NAMES)
MAX_EPOCHS = 20
BATCH_SIZE = 256

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'PQNet':
        model = PQNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
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
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (_, image, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                _, _, var_output = model(var_image)
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
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

    CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
    #torch.save(model.state_dict(), CKPT_PATH)
    torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
    print(' Pre-trianed model has been already save!')

def Test():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'PQNet':
        model = PQNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
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
    pred= torch.LongTensor().cuda()
    # switch to evaluate mode
    model.eval() #turn to test mode
    with torch.autograd.no_grad():
        for batch_idx, (_, image, label) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            _, var_output = model(var_image)
            var_output = F.log_softmax(var_output,dim=1) 
            var_output = var_output.max(1,keepdim=True)[1]
            pred = torch.cat((pred, var_output.data), 0)
            gt = torch.cat((gt, label.cuda()), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #evaluation
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    #F1 = 2 * (precision * recall) / (precision + recall)
    f1score = f1_score(gt_np, pred_np, average='micro')
    print('\r F1 Score = {:.4f}'.format(f1score))
    #sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
    sen = tp /(tp+fn)
    spe = tn /(tn+fp)
    print('\rSensitivity = {:.4f} and specificity = {:.4f}'.format(sen, spe))

def main():
    Train() #for training
    #Test() #for test

if __name__ == '__main__':
    main()