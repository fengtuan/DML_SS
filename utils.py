#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:43:28 2020

@author: weiyang
"""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import itertools
import subprocess
import os
import torch.nn.functional as F
def iterate_minibatches(ProteinLists, batchsize,shuffle=True):
    num_features=ProteinLists[0].Profile.shape[1]
    indices = np.arange(len(ProteinLists))
    if shuffle:        
        np.random.shuffle(indices)   
    maxLength=0
    inputs=torch.zeros(size=(batchsize,4096,num_features),dtype=torch.float32)
    masks=torch.zeros(size=(batchsize,4096),dtype=torch.bool)
    targets=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    for idx in range(len(ProteinLists)):
        if idx % batchsize==0:
            inputs.fill_(0)
            masks.fill_(False)
            targets.fill_(0)
            batch_idx=0          
            maxLength=0
        length=ProteinLists[indices[idx]].ProteinLen        
        masks[batch_idx,:length]=True
        inputs[batch_idx,:length,:]=ProteinLists[indices[idx]].Profile[:,:]
        targets[batch_idx,:length]=ProteinLists[indices[idx]].SecondarySeq[:]
        batch_idx+=1
        if length>maxLength:
                maxLength=length
        if (idx+1) % batchsize==0:
            yield inputs[:,:maxLength,:].transpose(1,2),targets[:,:maxLength],masks[:,:maxLength]
    if len(ProteinLists) % batchsize!=0:        
        yield inputs[:,:maxLength,:].transpose(1,2),targets[:,:maxLength],masks[:,:maxLength]


def train(args,model,device,train_list,optimizer,criterion):
    model.train()
    total_loss=0.0
    count=0
    for batch in iterate_minibatches(train_list,args.batch_size, shuffle=True):
        inputs, targets,masks = batch
        inputs=inputs.to(device)
        targets=targets.to(device)
        masks=masks.to(device)
        optimizer.zero_grad()
        outputs=model(inputs,masks)
        outputs=outputs.view(-1,outputs.size(2))
        loss=criterion(outputs,targets,masks)       
        total_loss += loss.item()
        count+=1
        loss.backward()
        optimizer.step()
    return total_loss/count 

def get_embedding_data(args,model,centroids,device,data_list,batch_size=64):
    model.eval()
    features_=[]
    labels_=[]
    prob_=[]
    with torch.no_grad():
        for batch in iterate_minibatches(data_list,batch_size, shuffle=False):
            inputs, targets,masks = batch           
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)            
            outputs=model(inputs,masks) 
            outputs=outputs.view(-1,outputs.size(2))
            
            targets=targets.flatten()
            masks=masks.flatten()
            outputs=torch.masked_select(outputs,masks.unsqueeze(1)).view(-1,outputs.shape[1])
            labels=torch.masked_select(targets,masks)
            #delete samples with 'X' as label           
            valid_idxs=(labels<args.num_class).nonzero(as_tuple=True)           
            labels=labels[valid_idxs]
            outputs=outputs[valid_idxs]
            
            S=torch.matmul(outputs,centroids.t())
            prob=F.softmax(args.tau*S,dim=-1)                      
            prob=prob[range(prob.shape[0]),labels]
            
            prob_.append(prob)
            features_.append(outputs)
            labels_.append(labels)
    features=torch.cat(features_,dim=0)
    labels=torch.cat(labels_,dim=0)
    prob_labels=torch.cat(prob_,dim=0)
    return features,labels,prob_labels



def Compute_Mean_Std(train_list):
    N=0.0
    d=train_list[0].Profile.shape[1]
    mean=torch.zeros(size=(d,),dtype=torch.float64)
    std=torch.zeros(size=(d,),dtype=torch.float64)
    for idx in range(len(train_list)):        
        N+=train_list[idx].ProteinLen
        mean+=train_list[idx].Profile.sum(dim=0)
        std+=train_list[idx].Profile.pow(2).sum(dim=0)
    mean=mean/N
    std=torch.sqrt((std-N*mean.pow(2))/(N-1))
    return mean.type(torch.float32).view(1,-1),std.type(torch.float32).view(1,-1)

def Normalized(data_list,mean,std):
    for idx in range(len(data_list)):
        data_list[idx].Profile=(data_list[idx].Profile-mean)/std



# Each class is represnted by its centroid, with test samples classified to the 
#     class with the nearest centroid.    
# def NearestCentroidClassifier(x,centroids):        
#     assert x.size()[1]==centroids.size()[1],"The dimension must match."
#     with torch.no_grad():
#         dist=1.0-torch.matmul(x,centroids.t())
#         idx=torch.topk(dist,1,dim=1,largest=False)[1]
#         pred_labels=idx.squeeze()
#     return pred_labels
def NearestCentroidClassifier(x,centroids):        
    assert x.size()[1]==centroids.size()[1],"The dimension must match."
    with torch.no_grad():
        dist=torch.matmul(x,centroids.t())
        idx=torch.topk(dist,1,dim=1,largest=True)[1]
        pred_labels=idx.squeeze()
    return pred_labels      

def eval(args,model,device,eval_list,batch_size,criterion=None):
    model.eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    with torch.no_grad():
        if criterion is not None:
            centroids=criterion.getNormalizedCentrioids()         
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)            
            outputs=model(inputs,masks)            
            if criterion is None:
                pred_labels=torch.argmax(outputs, dim=2)
            else:
                outputs=outputs.view(-1,outputs.size(2))
                pred_labels=NearestCentroidClassifier(outputs,centroids).view(batch_size,-1)
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]
                if L>0: 
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))     
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return accuracy


def eval_sov(args,model,device,eval_list,batch_size,criterion=None):
    model.eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    Eval_FileName=args.res_dir+'/'+args.dataset
    f = open(Eval_FileName+'_Q%d.txt'%(args.num_class), 'w')
    count=0
    if args.num_class==8:
       SS_dict={0:'L', 1:'B', 2:'E', 3:'G', 4:'I', 5:'H', 6:'S', 7:'T',8:'X'}
    else:
       SS_dict={0:'C', 1:'E', 2:'H',3:'X'} 
    with torch.no_grad():
        if criterion is not None:
            centroids=criterion.getNormalizedCentrioids()        
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)            
            outputs=model(inputs,masks)           
            if criterion is None:
                pred_labels=torch.argmax(outputs, dim=2)
            else:
                outputs=outputs.view(-1,outputs.size(2))
                pred_labels=NearestCentroidClassifier(outputs,centroids).view(batch_size,-1)
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]
                if L>0:
                    label_t=targets[i,:L].cpu().numpy()
                    label_p=pred_labels[i,:L].cpu().numpy()
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
                    label_t_=''                    
                    for i in label_t:                        
                        label_t_+=SS_dict[i] 
                    label_p_=''
                    for i in label_p:
                        label_p_+=SS_dict[i]
                   
                    f.write('>%s %d\n'%(eval_list[count].ProteinID,eval_list[count].ProteinLen))
                    f.write('%s\n'%(label_t_))
                    f.write('%s\n'%(label_p_))
                    count+=1
    
    f.close()
    commands="perl SOV.pl "+Eval_FileName+'_Q%d.txt'%(args.num_class)    
    subprocess.call(commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)   
    f=open(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class),'rt')
    line=f.readline()
    sov_results=line.strip() 
    f.close()          
    if os.path.exists(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)):
      os.remove(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)) 
    idxs=np.where(labels_true!=args.num_class)
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    F1=f1_score(labels_true,labels_pred,average='macro',labels=np.unique(labels_pred))
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))       
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return F1,accuracy,sov_results
def Ensemble_eval_sov(args,models_,device,eval_list,batch_size,centroids_):
    num_models=len(models_)    
    for l in range(num_models):
      models_[l].eval()
    labels_true=np.array([])
    labels_pred=np.array([])
    Eval_FileName=args.res_dir+'/'+args.dataset
    f = open(Eval_FileName+'_Q%d.txt'%(args.num_class), 'w')
    count=0
    if args.num_class==8:
       SS_dict={0:'L', 1:'B', 2:'E', 3:'G', 4:'I', 5:'H', 6:'S', 7:'T',8:'X'}
    else:
       SS_dict={0:'C', 1:'E', 2:'H',3:'X'} 
    with torch.no_grad():      
        for batch in iterate_minibatches(eval_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)
            masks=masks.to(device)
            dists_=[]
            for l in range(num_models):            
                outputs=models_[l](inputs,masks)     
                outputs=outputs.view(-1,outputs.size(2))
                S=torch.matmul(outputs,centroids_[l].t())
                # prob=F.softmax(args.tau*S,dim=-1)
                dists_.append(S)
                # dists_.append(prob)
            dist=dists_[0]
            for j in range(num_models-1):
                dist+=dists_[j+1]
            dist=dist/num_models            
            idx=torch.topk(dist,1,dim=1,largest=True)[1]
            pred_labels=idx.squeeze().view(batch_size,-1)            
            Lengths=masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L=Lengths[i]
                if L>0:
                    label_t=targets[i,:L].cpu().numpy()
                    label_p=pred_labels[i,:L].cpu().numpy()
                    labels_true=np.hstack((labels_true,targets[i,:L].cpu().numpy()))
                    labels_pred=np.hstack((labels_pred,pred_labels[i,:L].cpu().numpy()))
                    label_t_=''                    
                    for i in label_t:                        
                        label_t_+=SS_dict[i] 
                    label_p_=''
                    for i in label_p:
                        label_p_+=SS_dict[i]
                   
                    f.write('>%s %d\n'%(eval_list[count].ProteinID,eval_list[count].ProteinLen))
                    f.write('%s\n'%(label_t_))
                    f.write('%s\n'%(label_p_))
                    count+=1
    
    f.close()
    commands="perl SOV.pl "+Eval_FileName+'_Q%d.txt'%(args.num_class)    
    subprocess.call(commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)   
    f=open(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class),'rt')
    line=f.readline()
    sov_results=line.strip() 
    f.close()          
    if os.path.exists(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)):
      os.remove(Eval_FileName+'_Q%d_Eval.txt'%(args.num_class)) 
    idxs=np.where(labels_true!=args.num_class)
    labels_true=labels_true[idxs]
    labels_pred=labels_pred[idxs]
    F1=f1_score(labels_true,labels_pred,average='macro',labels=np.unique(labels_pred))
    class_correct=list(0. for i in range(args.num_class))
    class_total=list(0. for i in range(args.num_class))
    for i in range(len(labels_true)):
        label=int(labels_true[i])
        class_total[label]+=1
        if label==labels_pred[i]:
            class_correct[label]+=1
#    classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy=[]
    for i in range(args.num_class):
        accuracy.append(100.0*class_correct[i]/(class_total[i]+1e-12))       
    accuracy.append(100.0*sum(class_correct)/sum(class_total))
    return F1,accuracy,sov_results

def save_excel(eval_accuracy,FileName,SOV_Res):
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8') 
    sheet1 = workbook.add_sheet("result")    
    Header_8=['Q_L',"Q_B","Q_E","Q_G","Q_I","Q_H","Q_S","Q_T","Q₈","F₁-Score"]
    Header_3=['Q_C',"Q_E","Q_H","Q3","F₁-Score"]
    row=0
    if len(eval_accuracy)==len(Header_8):
        for i in range(len(Header_8)):
            sheet1.write(row,i,Header_8[i]) 
    else:
        for i in range(len(Header_3)):
            sheet1.write(row,i,Header_3[i])         
    row+=1
    for i in range(len(eval_accuracy)):
        sheet1.write(row,i,round(eval_accuracy[i],2))
    sheet2 = workbook.add_sheet("SOV")
    split_res=SOV_Res.split()
    Header2=[]
    eval_accuracy2=[]
    for i in range(len(split_res)):
        if i %2==0:
            Header2.append(split_res[i][:-1])
        else:
            eval_accuracy2.append(float(split_res[i]))    
    row=0
    for i in range(len(Header2)):
        sheet2.write(row,i,Header2[i])    
    row+=1
    for i in range(len(eval_accuracy2)):
        sheet2.write(row,i,eval_accuracy2[i])     
    workbook.save(r'%s.xls'%(FileName)) 
