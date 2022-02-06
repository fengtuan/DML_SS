import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

class SoftmaxLoss(torch.nn.Module):
    def __init__(self, num_classes, embed_dim, tau=18.0):
        super(SoftmaxLoss, self).__init__()        
        self.centroids = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        self.centroids.data.normal_(0, 1).renorm_(2, 1, 1e-5).mul_(1e5)        
        self.tau = tau 
       
    def getNormalizedCentrioids(self):
        P=F.normalize(self.centroids-self.centroids.mean(dim=-1,keepdim=True),2,dim=-1)
        return P

        
    def forward(self,x, targets,masks):
        targets=targets.flatten()
        masks=masks.flatten()
        X=torch.masked_select(x,masks.unsqueeze(1)).view(-1,x.shape[1])
        T=torch.masked_select(targets,masks)
        
        P=self.getNormalizedCentrioids()
        S=torch.matmul(X,P.t())
        T=F.one_hot(T,num_classes=self.centroids.shape[0])
        loss = torch.sum(- T * F.log_softmax(self.tau*S, -1), -1)
        loss = loss.mean()
        return loss   


class HardMarginLoss(torch.nn.Module):
    def __init__(self, num_classes, embed_dim,margin=0.15):
        super(HardMarginLoss, self).__init__()        
        self.centroids = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        self.centroids.data.normal_(0, 1).renorm_(2, 1, 1e-5).mul_(1e5) 
        self.margin=margin      

    def getNormalizedCentrioids(self):
        P=F.normalize(self.centroids-self.centroids.mean(dim=-1,keepdim=True),2,dim=-1)
        return P
        
    def forward(self,x, targets,masks):
        targets=targets.flatten()
        masks=masks.flatten()
        X=torch.masked_select(x,masks.unsqueeze(1)).view(-1,x.shape[1])
        T=torch.masked_select(targets,masks)
        n = len(T)
        
        P=self.getNormalizedCentrioids()
        Sim=torch.matmul(X,P.t())
        T=F.one_hot(T,num_classes=self.centroids.shape[0])
        pos_mask=T.type(torch.bool)
    
        neg_sim=torch.masked_select(Sim,pos_mask.bitwise_not()).view(n,-1)
        pos_sim=torch.masked_select(Sim,pos_mask).view(n,-1) 
        
        triple_loss =F.relu(neg_sim+self.margin-pos_sim).sum(dim=1,keepdim=True)       
        
        nonzero_cnt=float(torch.sum(triple_loss>0.0))
        loss=torch.sum(triple_loss)/nonzero_cnt
        
        return loss

   

class SoftMarginLoss(torch.nn.Module):
    def __init__(self, num_classes, embed_dim,beta=20.0):
        super(SoftMarginLoss, self).__init__()        
        self.centroids = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        self.centroids.data.normal_(0, 1).renorm_(2, 1, 1e-5).mul_(1e5) 
        self.beta=beta      

    def getNormalizedCentrioids(self):
        P=F.normalize(self.centroids-self.centroids.mean(dim=-1,keepdim=True),2,dim=-1)
        return P        
    def forward(self,x, targets,masks):
        targets=targets.flatten()
        masks=masks.flatten()
        X=torch.masked_select(x,masks.unsqueeze(1)).view(-1,x.shape[1])
        T=torch.masked_select(targets,masks)
        n = len(T)
        
        P=self.getNormalizedCentrioids()
        Sim=torch.matmul(X,P.t())
        T=F.one_hot(T,num_classes=self.centroids.shape[0])
        pos_mask=T.type(torch.bool)
    
        neg_sim=torch.masked_select(Sim,pos_mask.bitwise_not()).view(n,-1)
        pos_sim=torch.masked_select(Sim,pos_mask).view(n,-1) 
        
        soft_loss=F.softplus(neg_sim-pos_sim,self.beta).sum(dim=1,keepdim=True)
        
        nonzero_cnt=float(torch.sum(soft_loss>0.0))
        loss=torch.sum(soft_loss)/nonzero_cnt        
        return loss





