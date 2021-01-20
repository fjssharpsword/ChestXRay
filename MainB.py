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
import pandas as pd
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
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import heapq
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#self-defined
from CVTECXR import get_train_dataloader, get_test_dataloader
from Models.PQNet import PQNet

#command parameters
parser = argparse.ArgumentParser(description='For CVTE DR dataset')
parser.add_argument('--model', type=str, default='PQNet', help='PQNet')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
CLASS_NAMES = ['Negative', 'Positive']
N_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 64
MAX_EPOCHS = 100
SIM_THRESHOLD = 0.70
NUM_CLUSTERS = 32 #5

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'PQNet':
        model = PQNet(grid_vector = 128, is_pre_trained=True).cuda()#initialize model 
        #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer , step_size = 10, gamma = 1)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    me_criterion = nn.MSELoss() #regression loss function #nn.BCELoss()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    min_loss = 1.0 #float("inf") 
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (_, image, _) in enumerate(dataloader_train):
                #forward
                optimizer.zero_grad()
                var_image = torch.autograd.Variable(image).cuda()
                _, var_output = model(var_image)
                #backward
                loss_tensor = me_criterion(var_output, var_image)
                loss_tensor.backward() 
                optimizer.step()
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()       
                train_loss.append(loss_tensor.item()) 
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        if np.mean(train_loss) < min_loss:
        #if True:
            min_loss = np.mean(train_loss)
            CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
            torch.save(model.state_dict(), CKPT_PATH)
            #torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def PQTest():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    dataloader_test = get_test_dataloader(batch_size=1, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'PQNet':
        model = PQNet(grid_vector = 128, is_pre_trained=True).cuda()#initialize model 
        CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> loaded model checkpoint: "+ CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval() #turn to test mode
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=10)
    tsne = TSNE(n_components=2, init='pca', random_state=11)
    print('********************load model succeed!********************')

    print('********************begin production quantization!********************')
    #extract features
    PQVec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (_, image, _) in enumerate(dataloader_train):
            # convolutional features
            var_image = torch.autograd.Variable(image).cuda()
            var_vec, _  = model(var_image) 
            PQVec = torch.cat((PQVec, var_vec.data), 0)
            sys.stdout.write('\r training set process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #build codebook
    #bz*128*49, each image is segmented into 49 grids, each grid is respresented 128 dimensions.
    PQVec_np = PQVec.cpu().numpy() 
    PQCodebook = [] 
    for i in range(PQVec_np.shape[2]):
        roi_feat = PQVec_np[:,:,i].squeeze()
        kmeans.fit(roi_feat)
        PQCodebook.append(kmeans.cluster_centers_) 
        """
        #t-sne visualizing the effectiveness of clustering
        #r1 = pd.Series(kmeans.labels_).value_counts() #number of each cluster
        #r2 = pd.DataFrame(kmeans.cluster_centers_) #cluseter centroid
        #r = pd.concat([r2, r1], axis = 1) 
        #r = pd.concat([roi_feat, pd.Series(kmeans.labels_, index = roi_feat.index)], axis = 1)
        x_tsne = tsne.fit_transform(roi_feat)
        ScatterPlot(x_tsne, pd.Series(kmeans.labels_), str(i+1))
        """
        sys.stdout.write('\r codebook buliding process: = {}'.format(i+1))
        sys.stdout.flush()
    PQCodebook_np = np.array(PQCodebook) #49*NUM_CLUSTERS*128
    
    #test
    gt = torch.FloatTensor()
    pred= []
    with torch.autograd.no_grad():
        for batch_idx, (_, image, label) in enumerate(dataloader_test):
            gt = torch.cat((gt, label), 0)
            # convolutional features
            var_image = torch.autograd.Variable(image).cuda()
            var_vec, _  = model(var_image) 
            var_vec = var_vec.cpu().numpy()#1*128*49
            sim_com = []
            for i in range(PQCodebook_np.shape[0]):
                grid_vec = var_vec[:,:,i] #1*128
                grid_centroid = PQCodebook_np[i,:,:] #NUM_CLUSTERS*128
                sim_mat = cosine_similarity(grid_vec, grid_centroid) #1-paired_distances(grid_vec, grid_centroid)
                sim_com.append(np.max(sim_mat)) #most similarity
                #sim_com.append(np.mean(sim_mat))
            if np.min(sim_com) > SIM_THRESHOLD: 
                pred.append(0.0) #normal
            else:
                pred.append(1.0) #abnormaly
            sys.stdout.write('\r test set process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    #evaluation
    pred_np = np.array(pred)
    gt_np = gt.numpy()
    #F1 = 2 * (precision * recall) / (precision + recall)
    f1score = f1_score(gt_np, pred_np, average='micro')
    print('\r F1 Score = {:.4f}'.format(f1score))
    #sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(gt_np, pred_np).ravel()
    sen = tp /(tp+fn)
    spe = tn /(tn+fp)
    print('\rSensitivity = {:.4f} and specificity = {:.4f}'.format(sen, spe))  


def ScatterPlot(X, y, grid_idx):
    #X,y:numpy-array
    classes = len(list(set(y.tolist())))#get number of classes
    #palette = np.array(sns.color_palette("hls", classes))# choose a color palette with seaborn.
    color = ['c','y','m','b','g','r','w','k']
    marker = ['o','x','+','*','s','^','p','d']
    plt.figure(figsize=(8,8))#create a plot
    for i in range(classes):
        plt.scatter(X[y == i,0], X[y == i,1], c=color[i], marker=marker[i], label=str(i))
    plt.axis('off')
    plt.legend(loc='lower left')
    plt.savefig("/data/pycode/ChestXRay/Imgs/"+ grid_idx +'_tsne_vis.png', dpi=100) 

def main():
    #Train()
    PQTest() #for production quantization

if __name__ == '__main__':
    main()