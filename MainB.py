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
import heapq
from sklearn.cluster import KMeans
#self-defined
from CVTECXR import get_train_dataloader, get_test_dataloader
from Models.PQNet import PQNet

#command parameters
parser = argparse.ArgumentParser(description='For CVTE DR dataset')
parser.add_argument('--model', type=str, default='PQNet', help='PQNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
CLASS_NAMES = ['Negative', 'Positive']
N_CLASSES = len(CLASS_NAMES)
NUM_CLUSTERS = 256
BATCH_SIZE = 1024

def Cluster():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dataloader_test = get_test_dataloader(batch_size=1, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'PQNet':
        model = PQNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
        #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer , step_size = 10, gamma = 1)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval() #turn to test mode
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
    print('********************load model succeed!********************')

    print('********************begin production quantization!********************')
    #extract features
    feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (_, image, _) in enumerate(dataloader_train):
            var_image = torch.autograd.Variable(image).cuda()
            var_output = model(var_image).cpu()
            feat = torch.cat((feat, var_output), 0)
            #Bag of Visual Words
            """
            i = 0
            w = 8
            image = image.squeeze()
            vws = torch.FloatTensor()
            while (i + w <= image.size(1)):
                j = 0
                while (j + w <= image.size(2)):
                    vw = image[:, i:i+w, j:j+w]
                    vws = torch.cat((vws, vw.unsqueeze(0)), 0)
                    i = i+w
                    j = j+w
            vws = vws.view(vws.size(0)*vws.size(1), vws.size(2)*vws.size(3))
            feat = torch.cat((feat, vws.unsqueeze(0)), 0)
            """
            sys.stdout.write('\r training set process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
            break

    #build codebook
    feat_np = feat.numpy()
    codebook = [] 
    for i in range(feat_np.shape[1]):
        roi_feat = feat_np[:,i,:].squeeze()
        kmeans.fit(roi_feat)
        codebook.append(kmeans.cluster_centers_) 
        sys.stdout.write('\r codebood indexing process: = {}'.format(i+1))
        sys.stdout.flush()
    codebook_np = np.array(codebook)
    #test
    gt = torch.FloatTensor()
    pred= []
    with torch.autograd.no_grad():
        for batch_idx, (_, image, label) in enumerate(dataloader_test):
            gt = torch.cat((gt, label), 0)
            """
            i = 0
            w = 8
            image = image.squeeze()
            vws = torch.FloatTensor()
            while (i + w <= image.size(1)):
                j = 0
                while (j + w <= image.size(2)):
                    vw = image[:, i:i+w, j:j+w]
                    vws = torch.cat((vws, vw.unsqueeze(0)), 0)
                    i = i+w
                    j = j+w
            vws = vws.view(vws.size(0)*vws.size(1), vws.size(2)*vws.size(3)) 
            sim = []
            for i in range(codebook_np.shape[0]):
                te_feat = vws[i,:].reshape(1,vws.shape[1]) 
                roi_feat = codebook_np[i,:] 
                sim_mat = cosine_similarity(te_feat, roi_feat)
                sim.append(np.max(sim_mat))
            if np.mean(sim)<0.9: #abnormaly
                pred.append(1.0)
            else:
                pred.append(0.0) #normal
            """
            var_image = torch.autograd.Variable(image).cuda()
            var_output = model(var_image)
            var_output = var_output.cpu().numpy()
            sim = []
            for i in range(codebook_np.shape[0]):
                roi_feat = codebook_np[i,:] 
                te_feat = var_output[:,i,:] 
                sim_mat = cosine_similarity(te_feat, roi_feat)
                sim.append(np.max(sim_mat))
            if np.mean(sim)<0.9: #abnormaly
                pred.append(1.0)
            else:
                pred.append(0.0) #normal
            sys.stdout.write('\r test indexing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #evaluation
    pred_np = np.array(pred)
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
    Cluster() #for production quantization

if __name__ == '__main__':
    main()