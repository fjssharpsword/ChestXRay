import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import random

class TripletRankingLoss(nn.Module):
    def __init__(self, m=0.5):
        super(TripletRankingLoss, self).__init__()
        self.m = m 
    
    def forward(self, var_output_a, var_output_p, var_output_n):
        #input: batchsize*num_classes
        #output: loss
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_v = cos(var_output_a, var_output_p) - cos(var_output_a, var_output_n) + self.m
        loss = torch.where(cos_v<0, torch.zeros_like(cos_v), cos_v)

        return torch.mean(loss)

    """
    def _selTripletSamples(self, gt):
        #stime = time.time()
        bs = [x for x in range(gt.size(0))]#batch_size
        gt_sum = torch.sum(gt, 1)
        idxs = torch.where(gt_sum>0)[0]
        trSpl = [] #anchor, positive, semi-positive, negative
        for i in idxs:
            spl = [i, -1, -1, -1] #initialize an\pos\semipos\neg
            random.shuffle(bs)
            for j in bs:
                if spl[1]!=-1 and spl[2]!=-1 and spl[3]!=-1: break #stop search
                if i==j: continue #pass same sample
                if torch.equal(gt[i,:], gt[j,:]) == True: spl[1]=j #positive sample
                elif torch.mul(gt[i,:], gt[j,:]).sum() == 0: spl[3]=j #negative sample
                else: spl[2]=j #semipositive sample  
            if (spl[1]!=-1 and spl[2]!=-1) or  (spl[2]!=-1 and spl[3]!=-1) or (spl[1]!=-1 and spl[3]!=-1):
                trSpl.append(spl)
        #print("Triplet sampling: {} seconds".format(time.time()-stime))                 
        return trSpl
                        
    def forward(self, gt, pred):
        #gt: batchsize*num_classes
        #fea: batchsize*length of vector
        trSpl = self._selTripletSamples(gt)  #turn to similarity matrix
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        losses = [0.0]
        for spl in trSpl:
            a, p, s, n = spl[0], spl[1], spl[2], spl[3]
            if(p!=-1) and (s!=-1): 
                loss_ps = max(cos(pred[a,:], pred[p,:]) - cos(pred[a,:], pred[s,:]) + self.m_ps, 0)
                losses.append(loss_ps)
            if(s!=-1) and (n!=-1): 
                loss_sn = max(cos(pred[a,:], pred[s,:]) - cos(pred[a,:], pred[n,:]) + self.m_sn, 0)
                losses.append(loss_sn)
            if(p!=-1) and (n!=-1): 
                loss_pn = max(cos(pred[a,:], pred[p,:]) - cos(pred[a,:], pred[n,:]) + self.m_ps, 0)
                losses.append(loss_pn)
        return torch.mean(torch.tensor(losses))
    """
    """
    def _selTripletSamples(self, gt):
        stime = time.time()
        idxs_a = torch.where(torch.sum(gt, 1)>0)[0] #disease sample as anchor
        idxs_n = torch.where(torch.sum(gt, 1)==0)[0] #normal sample as negative
        samples = [] #anchor, positive, negative
        for id_a in idxs_a:
            #negative
            id_n = random.sample(list(idxs_n), 1)[0]
            #positive
            cols = torch.where(gt[id_a]==1)[0]
            rows = torch.where(gt[:, cols]==1)[0]
            id_p = random.sample(list(rows), 1)[0]
            samples.append([id_a, id_p, id_n])
        print("Triplet Sampling: {} seconds".format(time.time()-stime))                 
        return samples
                        
    def forward(self, gt, pred):
        #gt: batchsize*num_classes
        #fea: batchsize*length of vector
        samples = self._selTripletSamples(gt)  #turn to similarity matrix
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        losses = []
        for spl in samples:
            var_output_a = pred[spl[0]]
            var_output_p = pred[spl[1]]
            var_output_n = pred[spl[2]]
            cos_v = cos(var_output_a, var_output_p) - cos(var_output_a, var_output_n) + self.m
            loss = torch.where(cos_v<0, torch.zeros_like(cos_v), cos_v)
            losses.append(loss)
        return torch.mean(torch.tensor(losses))
    """
"""
#for debug   
gt = torch.zeros(512, 14)
pred = torch.rand(512, 14)
for i in range(250):#generate 0,1 randomly
    row = random.randint(0,511)
    ones_n = random.randint(1,2)
    col = [random.randint(0,13) for _ in range(ones_n)]
    gt[row, col] = 1
#a = torch.rand(512, 14)
#p = torch.rand(512, 14)
#n = torch.rand(512, 14)
trl = TripletRankingLoss()
loss = trl(gt,pred)
print(loss)
"""
