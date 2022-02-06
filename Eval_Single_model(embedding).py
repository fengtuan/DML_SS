#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 11:36:20 2021

@author: weiyang
"""
import torch
from networks import ConvNet_SS_embed
from  datasets import *
import sys
import os
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import utils
import time
import numpy as np
from loss import HardMarginLoss,SoftMarginLoss,SoftmaxLoss
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset',type=str,default='TEST2018')
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--proj_dim', default = 32, type = int)
parser.add_argument('--tau', default = 18.0, type = float)
parser.add_argument('--m', default = 0.15, type = float)
parser.add_argument('--beta', default = 20.0, type = float)
parser.add_argument('--loss_type',type=str,default="softmax")
parser.add_argument('--model_path',type=str,default="models")
parser.add_argument('--res_dir',type=str,default="Results")
parser.add_argument('--num_class',type=int,default=3)
parser.add_argument('--depth',type=int,default=2)
parser.add_argument('--num_channels',type=int,default=1024)
parser.add_argument('--dropout', default = 0.2, type = float)
args = parser.parse_args()

use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
  
if(os.path.isdir(args.res_dir)==False):
  os.mkdir(args.res_dir) 

if args.num_class==8:
    isEightClass=True
else:
    isEightClass=False

test2018_list = get_Test2018_embedding(isEightClass)
num_features=test2018_list[0].Profile.shape[1]

model=ConvNet_SS_embed(num_features,args.num_channels,args.depth,args.proj_dim,args.dropout).to(device)
if args.loss_type=="HML":
    criterion = HardMarginLoss(args.num_class,args.proj_dim,args.m).to(device)
elif args.loss_type=="SML":
    criterion = SoftMarginLoss(args.num_class,args.proj_dim,args.beta).to(device)
else:
    criterion = SoftmaxLoss(args.num_class,args.proj_dim,args.tau).to(device) 

# Load model
checkpoint=torch.load('%s/best_model_%d_%d_%d_%d_%d.pth'%(args.model_path,args.num_channels,args.proj_dim,args.depth,args.num_class,0))
model.load_state_dict(checkpoint['model'])
criterion.load_state_dict(checkpoint['criterion'])
#Evaluation Test2018
FileName=args.res_dir+'/'+str(args.proj_dim)+'_'+args.loss_type+'_'+str(args.num_channels)+'_'+str(args.depth)+'_'+str(args.num_class)+'_Test2018'
f = open(FileName+'.txt', 'w')
F1,accuracy,sov_results=utils.eval_sov(args,model,device,test2018_list,32,criterion)
if args.num_class==8:
    print('Q8: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    f.write('Q8:%.2f %%, F1:%.2f %%:\n'%(accuracy[-1],100*F1))
    f.write('  L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    f.write('  [%.2f,%.2f,%.2f, %.2f,%.2f,%.2f, %.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
else:
    print('Q3: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('Q3:%.2f %%, F1:%.2f %%:\n'%(accuracy[-1],100*F1))
    f.write('  C: %.2f,E: %.2f,H: %.2f\n'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('  [%.2f,%.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2]))   
f.close()
accuracy.append(100*F1)
utils.save_excel(accuracy,FileName,sov_results)