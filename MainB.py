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
from sklearn.metrics.pairwise import cosine_similarity
#self-defined
from CVTECXR import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from Utils.Evaluation import compute_AUCs, compute_fusion
#from Models.CVTEDRNet import CVTEDRNet
from Models.ASH import ASH, HashLossFunc

#command parameters
parser = argparse.ArgumentParser(description='For CVTE DR dataset')
parser.add_argument('--model', type=str, default='CVTEDRNet', help='CVTEDRNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "6"#"0,1,2,3,4,5"
CLASS_NAMES = ['Negative', 'Positive']
N_CLASSES = len(CLASS_NAMES)
MAX_EPOCHS = 20
BATCH_SIZE = 32 #128

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'ASH':
        model = ASH(code_size=1024, is_pre_trained=True).cuda()
        #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer , step_size = 10, gamma = 1)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    hash_criterion = HashLossFunc(margin=0.5)
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    best_loss = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image_a, image_b, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image_a = torch.autograd.Variable(image_a).cuda()
                var_image_b = torch.autograd.Variable(image_b).cuda()
                var_label = torch.autograd.Variable(label).cuda()

                var_output_a = model(var_image_a)
                var_output_b = model(var_image_b)
                loss_tensor = hash_criterion(var_output_a, var_output_b, var_label)
                #backward
                loss_tensor.backward() 
                optimizer.step()
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()       
                train_loss.append(loss_tensor.item()) 
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        if best_loss > np.mean(train_loss):
            best_loss = np.mean(train_loss)
            CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
            torch.save(model.state_dict(), CKPT_PATH)
            #torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_val = get_validation_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'ASH':
        model = ASH(code_size=1024, is_pre_trained=True).cuda()#initialize model 
        CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded Image model checkpoint: "+CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('******************** load model succeed!********************')

    print('******* begin indexing!*********')
    gt_val = torch.FloatTensor().cuda()
    feat_val = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_val):
            gt_val = torch.cat((gt_val, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_output = model(var_image)
            feat_val = torch.cat((feat_val, var_output.data), 0)

    gt_test =  torch.FloatTensor().cuda()
    feat_test = torch.FloatTensor().cuda()
    print('******* begin testing!*********')
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            gt_test = torch.cat((gt_test, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_output = model(var_image)
            feat_test = torch.cat((feat_test, var_output.data), 0)
    sim_mat = cosine_similarity(feat_val.cpu().numpy(), feat_test.cpu().numpy())
    pred_test = torch.FloatTensor().cuda()
    for i in range(sim_mat.shape[1]):
        idx = np.argmin(sim_mat[:,i])
        pred_test = torch.cat((pred_test, gt_val[idx].data), 0)

    sen, spe = compute_fusion(gt_test, pred_test)
    print('The Sensitivity is {:.4f} and the specificity is {:.4f}'.format(sen, spe))

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()