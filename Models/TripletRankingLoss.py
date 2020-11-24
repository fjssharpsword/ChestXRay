import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import random
from torch import Tensor
from typing import Tuple


class TripletRankingLoss(nn.Module):
    
    def __init__(self, m=0.1):
        super(TripletRankingLoss, self).__init__()
        self.m = m 
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    """
    #sampling one triplet samples for each sample
    def _selTripletSamples(self, pred:Tensor, gt:Tensor)-> Tuple[Tensor, Tensor, Tensor]:
        idxs_a = torch.where(torch.sum(gt, 1)>0)[0] #disease sample as anchor
        idxs_n = torch.where(torch.sum(gt, 1)==0)[0] #normal sample as negative
        anchor = torch.FloatTensor().cuda()
        positive = torch.FloatTensor().cuda()
        negative = torch.FloatTensor().cuda()
        for id_a in idxs_a:
            anchor = torch.cat((anchor, pred[id_a,:].unsqueeze(0).cuda()), 0)
            #positive
            cols = torch.where(gt[id_a]==1)[0]
            rows = torch.where(gt[:, cols]==1)[0]
            id_p = random.sample(list(rows), 1)[0]
            positive = torch.cat((positive, pred[id_p,:].unsqueeze(0).cuda()), 0)
            #negative
            idxs_a_cpu = idxs_a.cpu()
            idxs_n_cpu = idxs_n.cpu()
            rows = np.setdiff1d(idxs_a_cpu, rows.cpu()) #get other disease sample
            rows = np.union1d(rows, idxs_n_cpu) #not intersect1d
            id_n = random.sample(list(rows), 1)[0]
            negative = torch.cat((negative, pred[id_n,:].unsqueeze(0).cuda()), 0)
            
        for id_a in idxs_n:
            anchor = torch.cat((anchor, pred[id_a,:].unsqueeze(0).cuda()), 0)
            #positive
            id_p = random.sample(list(idxs_n), 1)[0]
            positive = torch.cat((positive, pred[id_p,:].unsqueeze(0).cuda()), 0)
            #nagative
            id_n = random.sample(list(idxs_a), 1)[0]
            negative = torch.cat((negative, pred[id_n,:].unsqueeze(0).cuda()), 0)

        return anchor, positive, negative
    """

    #sampling multi triplet samples for each anchor
    def _selTripletSamples(self, pred:Tensor, gt:Tensor)-> Tuple[Tensor, Tensor, Tensor]:
        idxs_a = torch.where(torch.sum(gt, 1)>0)[0] #disease sample as anchor
        idxs_n = torch.where(torch.sum(gt, 1)==0)[0] #normal sample as negative
        anchor = torch.FloatTensor().cuda()
        positive = torch.FloatTensor().cuda()
        negative = torch.FloatTensor().cuda()
        for id_a in idxs_a:
            cols = torch.where(gt[id_a]==1)[0]
            rows = torch.where(gt[:, cols]==1)[0]
            rows = np.unique(rows.cpu().numpy())
            pos_n = len(rows)
            for i in range(pos_n):
                anchor = torch.cat((anchor, pred[id_a,:].unsqueeze(0).cuda()), 0)
            positive = torch.cat((positive, pred[rows,:].cuda()), 0)
            #negative
            idxs_a_cpu = idxs_a.cpu()
            idxs_n_cpu = idxs_n.cpu()
            rows = np.setdiff1d(idxs_a_cpu, rows) #get other disease sample
            rows = np.union1d(rows, idxs_n_cpu) #not intersect1d
            id_n = random.sample(list(rows), pos_n)#[0]
            negative = torch.cat((negative, pred[id_n,:].cuda()), 0)
            
        for id_a in idxs_n:
            anchor = torch.cat((anchor, pred[id_a,:].unsqueeze(0).cuda()), 0)
            #positive
            id_p = random.sample(list(idxs_n), 1)[0]
            positive = torch.cat((positive, pred[id_p,:].unsqueeze(0).cuda()), 0)
            #nagative
            id_n = random.sample(list(idxs_a), 1)[0]
            negative = torch.cat((negative, pred[id_n,:].unsqueeze(0).cuda()), 0)

        return anchor, positive, negative

    def forward(self, pred, gt):
        #gt: batchsize*num_classes
        #fea: batchsize*length of vector
        anchor, positive, negative = self._selTripletSamples(pred, gt)  #turn to similarity matrix
        cos_v = self.cos(anchor, positive) - self.cos(anchor, negative) + self.m
        loss = torch.where(cos_v<0, torch.zeros_like(cos_v), cos_v)#max(cos_v, 0)
        loss = torch.mean(loss).requires_grad_()
        return loss
    
if __name__ == "__main__":
    #for debug   
    gt = torch.zeros(512, 14)
    pred = torch.rand(512, 64)
    for i in range(250):#generate 1 randomly
        row = random.randint(0,511)
        ones_n = random.randint(1,2)
        col = [random.randint(0,13) for _ in range(ones_n)]
        gt[i, col] = 1
    #a = torch.rand(512, 14)
    #p = torch.rand(512, 14)
    #n = torch.rand(512, 14)
    trl = TripletRankingLoss()
    loss = trl(pred,gt)
    print(loss)
