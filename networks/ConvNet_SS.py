#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:36:40 2021

@author: weiyang
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.hardswish(x)
        return x

class build_block(nn.Module):
    def __init__(self, in_channels, out_channels,dropout=0.2):
        super(build_block, self).__init__()       
        self.conv =BasicConv1d(in_channels, out_channels, kernel_size=1)
        self.branch_dropout = nn.Dropout(dropout)
        self.branch_conv =BasicConv1d(out_channels//2, out_channels//2, kernel_size=3)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):        
        out = self.conv(x)
        branch1,branch2 = out.chunk(2, dim=1)
        branch2 = self.branch_conv(self.branch_dropout(branch2))        
        out=torch.cat([branch1, branch2], 1)
        out=self.dropout(out)
        return out


class inception_unit(nn.Module):
    def __init__(self, in_channels,out_channels,dropout=0.2):
        super(inception_unit, self).__init__()       
        self.input_layers = nn.ModuleList(
            [build_block(in_channels,out_channels//2,dropout) for i in range(2)])
        self.intermediate_layer =build_block(out_channels//2,out_channels//2,dropout)
        

    def forward(self, x):
        branch1 = self.input_layers[0](x)
        branch2 = self.intermediate_layer(self.input_layers[1](x))       
        output = torch.cat([branch1, branch2], 1)      
        return output

def Normalize(x):    
    out=F.normalize(x-x.mean(dim=-1,keepdim=True),2,dim=-1)
    return out

class ConvNet_SS(nn.Module):
    def __init__(self,num_features,num_channels,depth=5,proj_dim=32,dropout=0.2):
        super(ConvNet_SS, self).__init__()
        assert(num_channels % 4==0)
        self.layer1 = inception_unit(num_features,num_channels,dropout)
        self.layer2 = nn.ModuleList(
            [inception_unit(num_channels,num_channels,dropout) for i in range(depth-1)]) 

        self.conv = nn.Conv1d(num_channels, 96, 9, padding=4)       
        self.fc = nn.Linear(96, proj_dim, bias=False)      
        self.depth=depth
        

    def forward(self, x,masks=None):
        output = self.layer1(x)
        for i in range(self.depth-1):
            output = self.layer2[i](output)        
        output = F.relu(self.conv(output)) 
        output = output.transpose(1,2)
        output = self.fc(output)        
        output=Normalize(output)
        return output

class ConvNet_SS_embed(nn.Module):
    def __init__(self,num_features,num_channels,depth=2,proj_dim=32,dropout=0.2):
        super(ConvNet_SS_embed, self).__init__()
        assert(num_channels % 4==0)
        self.layer1 = inception_unit(num_features,num_channels,dropout)
        self.layer2 = nn.ModuleList(
            [inception_unit(num_channels,num_channels,dropout) for i in range(depth-1)]) 

        self.conv = nn.Conv1d(num_channels, 96, 9, padding=4)       
        self.fc = nn.Linear(96, proj_dim, bias=False)      
        self.depth=depth
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x,masks=None):
        x = self.dropout(x)
        output = self.layer1(x)
        for i in range(self.depth-1):
            output = self.layer2[i](output)        
        output = F.relu(self.conv(output)) 
        output = output.transpose(1,2)
        output = self.fc(output)        
        output=Normalize(output)
        return output




if __name__=='__main__':
    model=ConvNet_SS(20,448)
    print(model)  

    
