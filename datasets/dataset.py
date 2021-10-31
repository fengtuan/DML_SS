# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 11:16:40 2017

@author: WeiYang
"""

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from enum import Enum
np.random.seed(0)
    
class ProteinNode:
    def __init__(self,ProteinID,ProteinLen,PrimarySeq,SecondarySeq,Profile):
        self.ProteinID=ProteinID
        self.ProteinLen=ProteinLen
        self.PrimarySeq=PrimarySeq        
        self.SecondarySeq=SecondarySeq
        self.Profile=Profile
 
def Load_H5PY(FilePath,isEightClass,dataset=None):
    SS=['L','B','E','G','I','H','S','T','X']
    SS8_Dict=dict(zip(SS,range(len(SS))))
    SS3_Dict={'L': 0, 'B': 1, 'E': 1, 'G': 2, 'I': 2, 'H': 2, 'S': 0, 'T': 0, 'X':3}
    Standard_AAS=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N','P','Q', 'R', 'S', 'T', 'V',  'W', 'Y']
    AA_Dict={}
    Non_Standard_AAS=list('BJOUXZ')
    for i,AA in enumerate(Standard_AAS):
        AA_Dict.update({AA:i})
    for i,AA in enumerate(Non_Standard_AAS):
        AA_Dict.update({AA:20})
    f = h5py.File(FilePath+'.h5', 'r')
    if dataset=='CB513':
        idxs=f['CB513_filtered_idxs'][()]
    elif dataset=='CASP12':
        idxs=f['CASP12_filtered_idxs'][()]           
    elif dataset=='CASP13':
        idxs=f['CASP13_filtered_idxs'][()]
    elif dataset=='CASP14':
        idxs=f['CASP14_filtered_idxs'][()]            
    else:
        NumOfSamples=len(f.keys())//4     
        idxs=range(NumOfSamples) 

    Data=[]
    for i in idxs:
        ProteinID=f['ID'+str(i)][()]
        #ProteinID=ProteinID.decode()
        PrimarySeq=f['PS'+str(i)][()]
        #PrimarySeq=PrimarySeq.decode()
        ProteinLen=len(PrimarySeq)
        PrimarySeq=[AA_Dict[e] for e in PrimarySeq]
        SecondarySeq=f['SS'+str(i)][()]
        #SecondarySeq=SecondarySeq.decode()
        if isEightClass:
            SecondarySeq=[SS8_Dict[e] for e in SecondarySeq]
        else:
            SecondarySeq=[SS3_Dict[e] for e in SecondarySeq]
        PrimarySeq=torch.tensor(PrimarySeq,dtype=torch.long)
        SecondarySeq=torch.tensor(SecondarySeq,dtype=torch.long)
        Profile=torch.from_numpy(f['Profile'+str(i)][()])        
        Node=ProteinNode(ProteinID,ProteinLen,PrimarySeq,SecondarySeq,Profile)            
        Data.append(Node)       
    f.close()
    return Data


def CB513(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CB513')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)   
   test_list=Load_H5PY('data/CB513',isEightClass)
   return train_list,valid_list,test_list

def CASP12(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',encodingType,dataset='CASP12')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP12',isEightClass)
   return train_list,valid_list,test_list

def CASP13(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CASP13')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP13',isEightClass)
   return train_list,valid_list,test_list

def CASP14(isEightClass):    
   Data=Load_H5PY('data/PISCES_Data',isEightClass,dataset='CASP14')
   train_list=Data[:-512] 
   valid_list=Data[-512:] 
   valid_list.sort(key=lambda Node: Node.ProteinLen,reverse=False)  
   test_list=Load_H5PY('data/CASP14',isEightClass)
   return train_list,valid_list,test_list

def Test2016_2018(isEightClass): 
   train_list=Load_H5PY('data/SPOT_1D_Train',isEightClass) 
   valid_list=Load_H5PY('data/SPOT_1D_Valid',isEightClass) 
   test2016_list=Load_H5PY('data/Test2016',isEightClass)
   test2018_list=Load_H5PY('data/Test2018',isEightClass)
   return train_list,valid_list,test2016_list,test2018_list

def get_Test2016(isEightClass):
   test2016_list=Load_H5PY('data/Test2016',isEightClass)
   return test2016_list

def get_Test2018(isEightClass):
   test2018_list=Load_H5PY('data/Test2018',isEightClass)
   return test2018_list



