
# Deep Metric Learning for Accurate Protein Secondary Structure Prediction



**Abstract**
<br>

Predicting the secondary structure of a protein from its amino acid sequence alone is a challenging prediction task for each residue in bioinformatics. Recent work has mainly used deep models based on the profile feature derived from multiple sequence alignments to make predictions. However, the existing state-of-the-art predictors usually have higher computational costs due to their large model sizes and complex network architectures. Here, we propose a simple yet effective deep centroid model for sequence-to-sequence secondary structure prediction based on deep metric learning. The proposed model adopts a lightweight embedding network with multibranch topology to map each residue in a protein chain into an embedding space. The goal of embedding learning is to maximize the similarity of each residue to its target centroid while minimizing its similarity to nontarget centroids. By assigning secondary structure types based on the learned centroids, we bypass the need for a time-consuming *k*-nearest neighbor search. Experimental results on six test sets demonstrate that our method achieves state-of-the-art performance with a simple architecture and smaller model size than existing models. Moreover, we also experimentally show that the embedding feature from the pretrained protein language model ProtT5-XL-U50 is superior to the profile feature in terms of prediction accuracy and feature generation speed. Code and datasets are available at https://github.com/fengtuan/DML_SS.


### 1. Datasets

The six test sets(CASP12, CASP13, CASP14, CB513, TEST 2016, TEST 2018) and their corresponding training and validation sets can be downloaded from Baidu Netdisk:

> * Profile-based hybrid feature data 

Baidu Netdisk:https://pan.baidu.com/s/1fwtekZH5zyodRj6WWPnWiA (Password: PSSP)  


> * The embedding feature data 
 
Baidu Netdisk:https://pan.baidu.com/s/1SMUOsSpsbJEEE99UbmuY5A (Password: PSSP)  

Please put the downloaded profile-based hybrid feature data files and embedding feature data files into the folders "data/hybrid" and "data/embedding" respectively. 

### 2. Requirement
> * Python >=3.8  
> * Pytorch >=1.10

The codes are tested under Ubuntu. 

### 3. Reproducibility
> * Eight class protein secondary structure prediction

python train_hybrid_feature.py --dataset 'CB513' --num_channels 448 --proj_dim 32 --depth 5 --num_class 8

python train_hybrid_feature.py --dataset 'CASP12' --num_channels 448 --proj_dim 32 --depth 5 --num_class 8

python train_hybrid_feature.py --dataset 'CASP13' --num_channels 448 --proj_dim 32 --depth 5 --num_class 8

python train_hybrid_feature.py --dataset 'CASP14' --num_channels 448 --proj_dim 32 --depth 5 --num_class 8

python train_hybrid_2016_2018.py --num_channels 448 --proj_dim 32 --depth 5 --num_class 8

python train_embedding_feature.py --dataset 'CB513' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 8

python train_embedding_feature.py --dataset 'CASP12' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 8

python train_embedding_feature.py --dataset 'CASP13' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 8

python train_embedding_feature.py --dataset 'CASP14' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 8

python train_embedding_2016_2018.py --num_channels 1024 --proj_dim 32 --depth 2 --num_class 8

> * Three class protein secondary structure prediction

python train_hybrid_feature.py --dataset 'CB513' --num_channels 448 --proj_dim 32 --depth 5 --num_class 3

python train_hybrid_feature.py --dataset 'CASP12' --num_channels 448 --proj_dim 32 --depth 5 --num_class 3

python train_hybrid_feature.py --dataset 'CASP13' --num_channels 448 --proj_dim 32 --depth 5 --num_class 3

python train_hybrid_feature.py --dataset 'CASP14' --num_channels 448 --proj_dim 32 --depth 5 --num_class 3

python train_hybrid_2016_2018.py --num_channels 448 --proj_dim 32 --depth 5 --num_class 3

python train_embedding_feature.py --dataset 'CB513' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 3

python train_embedding_feature.py --dataset 'CASP12' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 3

python train_embedding_feature.py --dataset 'CASP13' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 3

python train_embedding_feature.py --dataset 'CASP14' --num_channels 1024 --proj_dim 32 --depth 2 --num_class 3

python train_embedding_2016_2018.py --num_channels 1024 --proj_dim 32 --depth 2 --num_class 3

### 4. References
[1]Hanson, J., et al., Improving prediction of protein secondary structure, backbone angles, solvent accessibility and contact numbers by using predicted contact maps and an ensemble of recurrent and residual convolutional neural networks. Bioinformatics, 2019. 35(14): p. 2403-2410.

[2]Liu, T. and Z. Wang, SOV_refine: A further refined definition of segment overlap score and its significance for protein structure similarity. Source Code for Biology and Medicine, 2018. 13(1).


### 5. Citation
If you use our code in your study, please cite as:

[1] Wei Yang,Yang Liu, and Chunjing Xiao, Deep Metric Learning for Accurate Protein Secondary Structure Prediction,Knowledge-based systems,2022. 242: p. 108356.

### 6. Contact
For questions and comments, feel free to contact : yang0sun@gmail.com. 


