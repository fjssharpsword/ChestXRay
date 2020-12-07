# encoding: utf-8
"""
Training implementation
Author: Jason.Fang
Update time: 08/12/2020
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
 
#self-defined
from ChestXRay8 import get_train_dataloader, get_validation_dataloader, get_test_dataloader, get_bbox_dataloader
from Utils.Evaluation import compute_AUCs, compute_ROCCurve, compute_IoUs
from Utils.CAM import CAM
from Models.CVTEDRNet import CVTEDRNet

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

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dataloader_val = get_validation_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'CVTEDRNet':
        model = CVTEDRNet(num_classes=N_CLASSES, is_pre_trained=True).cuda()#initialize model 
    else: 
        print('No required model')
        return #over

    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    bce_criterion = nn.BCELoss() #define binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    AUROC_best = 0.50
    CKPT_PATH = '' #path for pre-trained model
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):  
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                optimizer.zero_grad()
                var_output = model(var_image)#forward
                loss_tensor = bce_criterion(var_output, var_label) 
                loss_tensor.backward() 
                optimizer.step()##update parameters    
                #print([x.grad for x in optimizer.param_groups[0]['params']])
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        model.eval()#turn to test mode
        val_loss = []
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_val):
                label = label.cuda()
                gt = torch.cat((gt, label), 0)
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_output = model(var_image)#forward
                pred = torch.cat((pred, var_output.data), 0)
                loss_tensor = bce_criterion(var_output, var_label) 
                sys.stdout.write('\r Epoch: {} / Step: {} : validation loss ={}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
                val_loss.append(loss_tensor.item())
        #evaluation       
        AUROCs = compute_AUCs(gt, pred, N_CLASSES)
        AUROC_avg = np.array(AUROCs).mean()
        print("\r Eopch: %5d validation loss = %.6f, Validataion AUROC = %.4f" % (epoch + 1, np.mean(val_loss), AUROC_avg)) 
        #save checkpoint
        if AUROC_best < AUROC_avg:
            AUROC_best = AUROC_avg
            CKPT_PATH = './Pre-trained/'+ args.model +'/best_model.pkl'
            #torch.save(model.state_dict(), CKPT_PATH)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        lr_scheduler_model.step()  #about lr and gamma
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
            var_output = model(var_image)#forward
            pred = torch.cat((pred, var_output.data), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
            
    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROCs[i]))
    print('The average AUROC is {:.4f}'.format(AUROC_avg))

    #Evaluating the threshold of prediction
    thresholds = compute_ROCCurve(gt, pred, CLASS_NAMES)
    return thresholds

def BoxTest(CKPT_PATH, thresholds):
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

    print('******* begin bounding box testing!*********')
    #predict bounding box
    #np.set_printoptions(suppress=True) #to float
    dataloader_bbox = get_bbox_dataloader(batch_size=1, shuffle=False, num_workers=0)
    cls_weights = list(model.parameters())
    #for name, layer in model.named_modules():
    #    print(name, layer)
    weight_softmax = np.squeeze(cls_weights[-3].data.cpu().numpy())
    cam = CAM(256, 224)
    IoUs =[]
    with torch.autograd.no_grad():
        for batch_idx, (_, gtbox, image, label) in enumerate(dataloader_bbox):
            var_image = torch.autograd.Variable(image).cuda()
            """
            var_output = model(var_image)#forward
            logit = var_output.cpu().data.numpy().squeeze() #predict
            idxs = []
            for i in range(N_CLASSES):
                if logit[i] > thresholds[i]: #diffrent diseases vary in threshold
                    idxs.append(i)
            """
            var_feature = model.dense_net_121.features(var_image) #get feature maps
            idx = torch.where(label[0]==1)[0] #true label
            cam_img = cam.returnCAM(var_feature.cpu().data.numpy(), weight_softmax, idx)
            pdbox = cam.returnBox(cam_img, gtbox[0].numpy())
            iou = compute_IoUs(pdbox, gtbox[0].numpy())
            IoUs.append(iou) #compute IoU
            if iou>0.9: 
                cam.visHeatmap(batch_idx, CLASS_NAMES[idx], image, cam_img, pdbox, gtbox[0].numpy()) #visulization
            sys.stdout.write('\r box process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    print('The average IoU is {:.4f}'.format(np.array(IoUs).mean()))

def main():
    CKPT_PATH = Train() #for training
    #CKPT_PATH ='./Pre-trained/'+ args.model +'/best_model.pkl' #for debug
    thresholds = Test(CKPT_PATH) #for test
    #thresholds = 0
    BoxTest(CKPT_PATH, thresholds)

if __name__ == '__main__':
    main()