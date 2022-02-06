#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:07:31 2021

@author: weiyang
"""
import torch
from networks import ConvNet_SS
from  datasets import *
import sys
import os
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import utils
import time
import numpy as np
from loss import SoftmaxLoss
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset',type=str,default='TEST2018')
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--proj_dim', default = 32, type = int)
parser.add_argument('--tau', default = 18.0, type = float)
parser.add_argument('--model_path',type=str,default="models")
parser.add_argument('--res_dir',type=str,default="Results")
parser.add_argument('--num_class',type=int,default=8)
parser.add_argument('--depth',type=int,default=2)
parser.add_argument('--num_channels',type=int,default=1024)
args = parser.parse_args()


use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if(os.path.isdir(args.res_dir)==False ):
  os.mkdir(args.res_dir)
if args.num_class==8:
    isEightClass=True
else:
    isEightClass=False


test2018_list = get_Test2018_embedding(isEightClass)
num_features=test2018_list[0].Profile.shape[1]
models_=[]
centroids_=[]
seeds=[0,10000,20000,30000,40000]
widths=[1024]
for l in range(len(seeds)):
    for i in range(len(widths)):
        model=ConvNet_SS(num_features,widths[i],args.depth,args.proj_dim).to(device)
        criterion = SoftmaxLoss(args.num_class,args.proj_dim,args.tau).to(device)
        checkpoint=torch.load('%s/best_model_%d_%d_%d_%d_%d.pth'%(args.model_path,widths[i],args.proj_dim,args.depth,args.num_class,seeds[l]))
        model.load_state_dict(checkpoint['model'])
        criterion.load_state_dict(checkpoint['criterion'])
        centroids=criterion.getNormalizedCentrioids() 
        models_.append(model)
        centroids_.append(centroids)

print(args.dataset)
F1,accuracy,sov_results=utils.Ensemble_eval_sov(args,models_,device,test2018_list,args.batch_size,centroids_)
if args.num_class==8:
    print('Q8: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
else:
    print('Q3: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
accuracy.append(100*F1)
FileName=args.res_dir+'/'+'Ensemble_2018_%d'%(args.num_class)
utils.save_excel(accuracy,FileName,sov_results) 