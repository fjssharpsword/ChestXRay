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
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from skimage.measure import label
from PIL import Image

from ChestXRay8 import get_train_dataloader, get_validation_dataloader, get_test_dataloader
from Utils import compute_AUCs
from Models.AGCNN import Densenet121_AG, Attention_gen_patchs, Fusion_Branch

#command parameters
parser = argparse.ArgumentParser(description='For ChestXRay')
parser.add_argument('--model', type=str, default='AGCNN', help='AGCNN')
args = parser.parse_args()

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = len(CLASS_NAMES)
MAX_EPOCHS = 30
BATCH_SIZE = 128

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=8) #get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    Global_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Global_Branch_model = nn.DataParallel(Global_Branch_model).cuda()  # make model available multi GPU cores training
    Local_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Local_Branch_model = nn.DataParallel(Local_Branch_model).cuda()  
    Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()
    Fusion_Branch_model = nn.DataParallel(Fusion_Branch_model).cuda()  

    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    criterion = nn.BCELoss()
    optimizer_global = optim.Adam(Global_Branch_model.parameters(), lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_global = lr_scheduler.StepLR(optimizer_global , step_size = 10, gamma = 1)
    
    optimizer_local = optim.Adam(Local_Branch_model.parameters(), lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_local = lr_scheduler.StepLR(optimizer_local , step_size = 10, gamma = 1)
    
    optimizer_fusion = optim.Adam(Fusion_Branch_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion , step_size = 15, gamma = 0.1)
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    CKPT_PATH_G = '' 
    CKPT_PATH_L = '' 
    CKPT_PATH_F = ''
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        Global_Branch_model.train()  #set model to training mode
        Local_Branch_model.train()
        Fusion_Branch_model.train()
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                optimizer_global.zero_grad()
                optimizer_local.zero_grad()
                optimizer_fusion.zero_grad()

                output_global, fm_global, pool_global = Global_Branch_model(var_image)
                patchs_var = Attention_gen_patchs(image,fm_global)
                output_local, _, pool_local = Local_Branch_model(patchs_var)
                output_fusion = Fusion_Branch_model(pool_global, pool_local)
                # loss
                loss_global = criterion(output_global, var_label)
                loss_local = criterion(output_local, var_label)
                loss_fusion = criterion(output_fusion, var_label)
                loss_tensor = loss_global*0.8 + loss_local*0.2 + loss_fusion*0.1
                loss_tensor.backward()
                #loss_global.backward()
                #loss_local.backward()
                #loss_fusion.backward()
                optimizer_global.step()  #UPDATE
                optimizer_local.step()
                optimizer_fusion.step()
                
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        AUROC_avg = Evaluation(Global_Branch_model, Local_Branch_model, Fusion_Branch_model,dataloader_val, type=False)
        print("\r Eopch: %5d Validataion AUROC = %.4f" % (epoch + 1, AUROC_avg)) 

        if AUROC_best < AUROC_avg:
            AUROC_best = AUROC_avg
            CKPT_PATH_G = './Pre-trained/'+ args.model +'/best_model_global.pkl'
            CKPT_PATH_L = './Pre-trained/'+ args.model +'/best_model_local.pkl'
            CKPT_PATH_F = './Pre-trained/'+ args.model +'/best_model_fusion.pkl'
            torch.save(Global_Branch_model.state_dict(), CKPT_PATH_G)
            torch.save(Local_Branch_model.state_dict(), CKPT_PATH_L)
            torch.save(Fusion_Branch_model.state_dict(), CKPT_PATH_F)
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        lr_scheduler_global.step()  #about lr and gamma
        lr_scheduler_local.step() 
        lr_scheduler_fusion.step() 
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
    return [CKPT_PATH_G, CKPT_PATH_L, CKPT_PATH_F]    

def Test(CKPT_PATH):
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    Global_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Local_Branch_model = Densenet121_AG(pretrained = False, num_classes = N_CLASSES).cuda()
    Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()

    Global_Branch_model = nn.DataParallel(Global_Branch_model).cuda()  # make model available multi GPU cores training
    Local_Branch_model = nn.DataParallel(Local_Branch_model).cuda()
    Fusion_Branch_model = nn.DataParallel(Fusion_Branch_model).cuda()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly

    CKPT_PATH_G = CKPT_PATH[0]
    CKPT_PATH_L = CKPT_PATH[1]
    CKPT_PATH_F = CKPT_PATH[2]
    if os.path.isfile(CKPT_PATH_G):
        checkpoint = torch.load(CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint" + CKPT_PATH_G)

    if os.path.isfile(CKPT_PATH_L):
        checkpoint = torch.load(CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint" + CKPT_PATH_L)

    if os.path.isfile(CKPT_PATH_F):
        checkpoint = torch.load(CKPT_PATH_F)
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint" + CKPT_PATH_F)   
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    AUROC_avg = Evaluation(Global_Branch_model, Local_Branch_model, Fusion_Branch_model,dataloader_test, type=True)
    print("\r The average AUROC of testset = %.4f" % (AUROC_avg)) 

def Evaluation(model_global, model_local, model_fusion, test_loader, type=False):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_global = torch.FloatTensor().cuda()
    pred_local = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_global.eval()
    model_local.eval()
    model_fusion.eval()
    cudnn.benchmark = True

    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():     
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())

            output_global, fm_global, pool_global = model_global(input_var)
            
            patchs_var = Attention_gen_patchs(inp,fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global,pool_local)

            pred_global = torch.cat((pred_global, output_global.data), 0)
            pred_local = torch.cat((pred_local, output_local.data), 0)
            pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
            
            sys.stdout.write('\r testing process: = {}'.format(i+1))
            sys.stdout.flush()

    if type == True: #test dataset
        AUROCs_g = compute_AUCs(gt, pred_global, N_CLASSES)
        AUROC_avg = np.array(AUROCs_g).mean()
        print('Global branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_g[i]))

        AUROCs_l = compute_AUCs(gt, pred_local)
        AUROC_avg = np.array(AUROCs_l).mean()
        print('\n')
        print('Local branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_l[i]))

        AUROCs_f = compute_AUCs(gt, pred_fusion)
        AUROC_avg = np.array(AUROCs_f).mean()
        print('\n')
        print('Fusion branch: The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs_f[i]))
        return AUROC_avg

    else: #validation dataset
        AUROCs_f = compute_AUCs(gt, pred_fusion)
        AUROC_avg = np.array(AUROCs_f).mean()
        return AUROC_avg

def main():
    CKPT_PATH = Train() #for training
    Test(CKPT_PATH) #for test

if __name__ == '__main__':
    main()