#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:44:46 2021

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
parser.add_argument('--dataset',type=str,default='Test2016_2018')
parser.add_argument('--encodingType', default = 'spot_1d_57')
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--maxEpochs', default =100, type = int)
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--proj_dim', default = 32, type = int)
parser.add_argument('--tau', default = 18.0, type = float)
parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3) 
parser.add_argument('--model_path',type=str,default="hybrid_2016_2018")
parser.add_argument('--res_dir',type=str,default="hybrid_2016_2018")
parser.add_argument('--num_class',type=int,default=8)
parser.add_argument('--depth',type=int,default=4)
parser.add_argument('--num_channels',type=int,default=448)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed) 
use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:     
    torch.cuda.manual_seed_all(args.seed)
if(os.path.isdir(args.model_path)==False ):
  os.mkdir(args.model_path)   
if(os.path.isdir(args.res_dir)==False ):
  os.mkdir(args.res_dir) 

print("Starting training...")
if args.num_class==8:
    isEightClass=True
else:
    isEightClass=False

if  args.encodingType=='spot_1d_57':  
    train_list,valid_list,test2016_list,test2018_list = Load_DataSet('Test2016_2018',EncodingType.spot_1d_57,isEightClass) 
else:        
    train_list,valid_list,test2016_list,test2018_list = Load_DataSet('Test2016_2018',EncodingType.spot_1d_73,isEightClass)



num_features=train_list[0].Profile.shape[1]
model=ConvNet_SS(num_features,args.num_channels,args.depth,args.proj_dim).to(device)
criterion = SoftmaxLoss(args.num_class,args.proj_dim,args.tau).to(device)
FileName=args.res_dir+'/'+str(args.proj_dim)+'_'+str(args.tau)+'_'+str(args.num_channels)+'_'+str(args.depth)+'_'+str(args.seed)+'_'+str(args.num_class)+'_'+time.strftime('_%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
checkpoint = {
    'model': model.state_dict(),  
    'criterion': criterion.state_dict()
}

param_dict={}
for k,v in model.named_parameters():
    param_dict[k]=v
bn_params=[v for n,v in param_dict.items() if ('bn' in n or 'bias' in n)]
rest_params=[v for n,v in param_dict.items() if not ('bn' in n or 'bias' in n)]
optimizer = torch.optim.AdamW([{'params':bn_params,'weight_decay':0},
                              {'params':rest_params,'weight_decay':args.weight_decay},
                                {'params':criterion.parameters(),'weight_decay':args.weight_decay,'lr':args.lr}],
                              lr=args.lr,amsgrad=False)
#############################################################
f = open(FileName+'.txt', 'w')
# early-stopping parameters
decrease_patience=8
best_accuracy = 0
best_epoch = 0
epoch = 0
Num_Of_decrease=0
done_looping = False  
  
while (epoch < args.maxEpochs) and (not done_looping):
    epoch = epoch + 1
    start_time = time.time()  
    average_loss=utils.train(args,model,device,train_list,optimizer,criterion)
    # scheduler.step()
    print("{}th Epoch took {:.3f}s".format(epoch, time.time() - start_time))
    f.write("{}th Epoch took {:.3f}s\n".format(epoch, time.time() - start_time))
    print("  training loss:\t\t{:.3f}".format(average_loss))
    f.write("  training loss:\t\t{:.3f}\n".format(average_loss))

  
    accuracy=utils.eval(args,model,device,valid_list,args.batch_size,criterion) 
    if args.num_class==8:
        print("  validation Q8 accuracy:\t\t{:.2f}".format(accuracy[-1]))
        f.write("  validation Q8 accuracy:\t\t{:.2f}\n".format(accuracy[-1]))
        print('  L: %.2f %%,B: %.2f %%,E: %.2f %%,G: %.2f %%,I: %.2f %%,H: %.2f %%,S: %.2f %%,T: %.2f %% '%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
        f.write('[%.2f %%,%.2f %%,%.2f %%, %.2f %%,%.2f %%,%.2f %%, %.2f %%,%.2f %%] \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    else:
        print("  validation Q3 accuracy:\t\t{:.2f}".format(accuracy[-1]))
        f.write("  validation Q3 accuracy:\t\t{:.2f}\n".format(accuracy[-1]))
        print('  C: %.2f %%,E: %.2f %%,H: %.2f %%'%(accuracy[0],accuracy[1],accuracy[2]))
        f.write('[%.2f %%,%.2f %%,%.2f %%] \n'%(accuracy[0],accuracy[1],accuracy[2]))
 
    f.flush()
    # if we got the best validation accuracy until now
    if accuracy[-1] > best_accuracy:
        best_accuracy = accuracy[-1]
        best_epoch = epoch
        Num_Of_decrease=0        
        torch.save(checkpoint,'%s/best_model_%d_%d_%d_%d_%d.pth'%(args.model_path,args.num_channels,args.proj_dim,args.depth,args.num_class,args.seed))
    else:
        Num_Of_decrease=Num_Of_decrease+1
    if (Num_Of_decrease>decrease_patience):
        done_looping = True    
print('The validation accuracy %.2f %% of the best model in the %i th epoch' 
            %(best_accuracy,best_epoch))
f.write('The validation accuracy %.2f %% of the best model in the %i th epoch\n' 
            %(best_accuracy,best_epoch))
f.close()


# Load model
checkpoint=torch.load('%s/best_model_%d_%d_%d_%d_%d.pth'%(args.model_path,args.num_channels,args.proj_dim,args.depth,args.num_class,args.seed))
model.load_state_dict(checkpoint['model'])
criterion.load_state_dict(checkpoint['criterion'])
#Evaluation Test2016
FileName=args.res_dir+'/'+str(args.proj_dim)+'_'+str(args.tau)+'_'+str(args.num_channels)+'_'+str(args.depth)+'_'+str(args.num_class)+'_Test2016'+'_'+str(args.seed)
f = open(FileName+'.txt', 'w')
F1,accuracy,sov_results=utils.eval_sov(args,model,device,test2016_list,32,criterion)
if args.num_class==8:
    print('Q8: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    print('8-state SOV results:')
    print(sov_results)
    f.write('Q8:%.2f %%, F1:%.2f %%:\n'%(accuracy[-1],100*F1))
    f.write('  L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    f.write('  [%.2f,%.2f,%.2f, %.2f,%.2f,%.2f, %.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    f.write('8-state SOV results:\n')
    f.write(' %s\n'%(sov_results))
else:
    print('Q3: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
    print('3-state SOV results:')
    print(sov_results)
    f.write('Q3:%.2f %%, F1:%.2f %%:\n'%(accuracy[-1],100*F1))
    f.write('  C: %.2f,E: %.2f,H: %.2f\n'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('  [%.2f,%.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('3-state SOV results:\n')
    f.write(' %s\n'%(sov_results))    
f.close()
accuracy.append(100*F1)
utils.save_excel(accuracy,FileName,sov_results)
#Evaluation Test2018
FileName=args.res_dir+'/'+str(args.proj_dim)+'_'+str(args.tau)+'_'+str(args.num_channels)+'_'+str(args.depth)+'_'+str(args.num_class)+'_Test2018'+'_'+str(args.seed)
f = open(FileName+'.txt', 'w')
F1,accuracy,sov_results=utils.eval_sov(args,model,device,test2018_list,32,criterion)
if args.num_class==8:
    print('Q8: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    print('8-state SOV results:')
    print(sov_results)
    f.write('Q8:%.2f %%, F1:%.2f %%:\n'%(accuracy[-1],100*F1))
    f.write('  L: %.2f,B: %.2f,E: %.2f,G: %.2f,I: %.2f,H: %.2f,S: %.2f,T: %.2f \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    f.write('  [%.2f,%.2f,%.2f, %.2f,%.2f,%.2f, %.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
    f.write('8-state SOV results:\n')
    f.write(' %s\n'%(sov_results))
else:
    print('Q3: %.2f %%, F1: %.2f %%  on the dataset  %s'%(accuracy[-1],100*F1,args.dataset))
    print('C: %.2f,E: %.2f,H: %.2f'%(accuracy[0],accuracy[1],accuracy[2]))
    print('3-state SOV results:')
    print(sov_results)
    f.write('Q3:%.2f %%, F1:%.2f %%:\n'%(accuracy[-1],100*F1))
    f.write('  C: %.2f,E: %.2f,H: %.2f\n'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('  [%.2f,%.2f,%.2f] \n'%(accuracy[0],accuracy[1],accuracy[2]))
    f.write('3-state SOV results:\n')
    f.write(' %s\n'%(sov_results))    
f.close()
accuracy.append(100*F1)
utils.save_excel(accuracy,FileName,sov_results)
