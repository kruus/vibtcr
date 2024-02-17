#!/usr/bin/env python
# coding: utf-8

# # Attentive Variational Information Bottleneck
# 
# In this notebook, we train and test the Attentive Variational Information Bottleneck (MVIB [1] with Attention of Experts) and MVIB on all datasets.
# 
# [1] Microbiome-based disease prediction with multimodal variational information bottlenecks, Grazioli et al., https://www.biorxiv.org/node/2109522.external-links.html

# In[1]:


import pandas as pd
import torch
import numpy as np
import random

from vibtcr.dataset import TCRDataset
from vibtcr.mvib.mvib import MVIB
from vibtcr.mvib.mvib_trainer import TrainerMVIB

from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[2]:


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
import pandas as pd
import torch

metrics = [
    'auROC',
    'Accuracy',
    'Recall',
    'Precision',
    'F1 score',
    'auPRC'
]

def pr_auc(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return pr_auc

def get_scores(y_true, y_prob, y_pred):
    """
    Compute a df with all classification metrics and respective scores.
    """
    
    scores = [
        roc_auc_score(y_true, y_prob),
        accuracy_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        pr_auc(y_true, y_prob)
    ]
    
    df = pd.DataFrame(data={'score': scores, 'metrics': metrics})
    return df


# In[3]:


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


# In[4]:


import os
login = os.getlogin( )
DATA_BASE = f"/home/{login}/Git/tcr/data/"
RESULTS_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/results/"
# To run in github checkout of vibtcr, after `unzip data.zip` ...
RESULTS_BASE = os.path.join('.', 'results')
DATA_BASE = os.path.join('..', '..', 'data')


# In[5]:


#device = torch.device('cuda:0')

batch_size = 4096
epochs = 500 #500 <-------------------------------------
lr = 1e-3

z_dim = 150
early_stopper_patience = 50
monitor = 'auROC'
lr_scheduler_param = 10

beta = 1e-6


# # AVIB multimodal pooling of experts (aoe)

# In[6]:


def train_generic(
    modality:str,
    joint_posterior:str,
    data_type:str,
    device:str = "cuda"
    folds:int = 5
):
    assert modality        in ['bimodal', 'trimodal', 'bimodal-alpha']
    assert joint_posterior in ['aoe', 'poe', 'max_pool', 'avg_pool']
    assert data_type       in ["alpha+beta-only", "full", "beta-only"]
    run_name_base = f"mvib.{modality}.{joint_posterior}.{data_type}" #.rep-{i}

    cdr3a_col = None
    cdr3b_col = None
    trainer_mode = modality
    if modality == 'bimodal':
        cdr3b_col='tcrb'
    elif modality == 'trimodal':
        cdr3a_col='tcra'
        cdr3b_col='tcrb'
    elif modality == 'bimodal-alpha':
        cdr3b_col = 'tcra'              # <- just pretend
        trainer_mode = 'bimodal'

    if data_type == "alpha+beta-only":
        df = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'alpha-beta.csv'))
    elif data_type == "full":
        df1 = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'beta.csv'))
        df2 = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'alpha-beta.csv'))
        df = pd.concat([df1, df2]).reset_index()
    else:
        assert data_type == "beta-only"
        df = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'beta.csv'))

    for i in range(folds):  # 'folds' independent train/test splits
        set_random_seed(i)

        df_train, df_test = train_test_split(df.copy(), test_size=0.2, random_state=i)
        scaler = TCRDataset(df_train.copy(), torch.device("cpu"), cdr3b_col=cdr3b_col, cdr3a_col=cdr3a_col).scaler

        ds_test = TCRDataset(df_test, torch.device("cpu"), cdr3b_col=cdr3b_col, cdr3a_col=cdr3a_col, scaler=scaler)

        df_train, df_val = train_test_split(df_train, test_size=0.2, stratify=df_train.sign, random_state=i)
            
        # train loader with balanced sampling
        ds_train = TCRDataset(df_train, device, cdr3b_col=cdr3b_col, cdr3a_col=cdr3a_col, scaler=scaler)
        class_count = np.array([df_train[df_train.sign == 0].shape[0], df_train[df_train.sign == 1].shape[0]])
        weight = 1. / class_count
        samples_weight = torch.tensor([weight[s] for s in df_train.sign])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = torch.utils.data.DataLoader( ds_train, batch_size=batch_size, sampler=sampler)
        
        # val loader with balanced sampling
        ds_val = TCRDataset(df_val, device, cdr3b_col=cdr3b_col, cdr3a_col=cdr3a_col, scaler=scaler)
        class_count = np.array([df_val[df_val.sign == 0].shape[0], df_val[df_val.sign == 1].shape[0]])
        weight = 1. / class_count
        samples_weight = torch.tensor([weight[s] for s in df_val.sign])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        val_loader = torch.utils.data.DataLoader( ds_val, batch_size=batch_size, sampler=sampler)

        model = MVIB(z_dim=z_dim, device=device, joint_posterior=joint_posterior).to(device)

        trainer = TrainerMVIB(
            model, epochs=epochs, lr=lr, beta=beta, checkpoint_dir=".",
            mode=trainer_mode, lr_scheduler_param=lr_scheduler_param
        )
        checkpoint = trainer.train(train_loader, val_loader, early_stopper_patience, monitor)
        run_name = f"{run_name_base}.rep-{i}"
        trainer.save_checkpoint(checkpoint, folder='./', filename=os.path.join(RESULTS_BASE, f"{run_name}.pth"))
        
        # test
        model = MVIB.from_checkpoint(checkpoint, torch.device("cpu"))
        model.train()
        model.to(device=device)
        with torch.no_grad():
            pred = model.classify(pep=ds_test.pep, cdr3b=ds_test.cdr3b, cdr3a=None)

        pred = pred.detach().numpy()
        df_test['prediction_'+str(i)] = pred.squeeze().tolist()

        # save results for further analysis
        df_test.to_csv(os.path.join(RESULTS_BASE, f"{run_name}.csv"), index=False)
        print(f"Good: {run_name} DONE!")

    print("GOOD: mvib.{run_name_base}.* FINISHED")

#
# NOTE: This notebook runs several choices for "joint_posterior"
#       Class BaseVIB supports:
#         'peo' ~ "product of experts" ~ "MVIB" (multimodal) original paper
#         'aoe' ~ "attention of experts" ~ "AVIB" (attentive)
#         'max_pool'
#         'avg_pool'
#joint_posterior = "aoe"

# # alpha+beta set - peptide+CDR3b (AVIB)
# In[7]:
train_generic("bimodal", "aoe", "alpha+beta-only")

# # alpha+beta set - peptide+CDR3b+CDR3a (AVIB)
# In[8]:
#modality = "trimodal"
#joint_posterior = "aoe"
#data_type = "alpha+beta-only"
train_generic("trimodal", "aoe", "alpha+beta-only")

# # alpha+beta set - peptide+CDR3a (AVIB)
# In[9]:
#modality = "bimodal-alpha"
#joint_posterior = "aoe"
#data_type = "alpha+beta-only"
train_generic("bimodal-alpha", "aoe", "alpha+beta-only")

# # beta set (AVIB)
# In[10]:
#modality = "bimodal"
#joint_posterior = "aoe"
#data_type = "beta-only"
train_generic("bimodal", "aoe", "beta-only")

# # full set: alpha+beta set + beta set (AVIB)
# In[11]:
#modality = "bimodal"
#joint_posterior = "aoe"
#data_type = "full"
train_generic("bimodal", "aoe", "full")

# # MVIB multimodal pooling of experts (poe)
# In[12]:
joint_posterior = "poe"
# # alpha+beta set - peptide+CDR3b (MVIB)
# In[13]:
#modality = "bimodal"
#joint_posterior = "poe"
#data_type = "alpha+beta-only"
train_generic("bimodal", "poe", "alpha+beta-only")

# # alpha+beta set - peptide+CDR3b+CDR3a (MVIB)
# In[14]:
#modality="trimodal"
#joint_posterior = "poe"
#data_type = "alpha+beta-only"
train_generic("trimodal", "poe", "alpha+beta-only")

# # alpha+beta set - peptide+CDR3a (MVIB)
# In[15]:
#modality = "bimodal-alpha"
#joint_posterior = "poe"
#data_type = "alpha+beta-only"
train_generic("bimodal-alpha", "poe", "alpha+beta-only")

# # beta set (MVIB)
# In[16]:
#modality = "bimodal"
#joint_posterior = "poe"
#data_type = "beta-only"
train_generic("bimodal", "poe", "beta-only")

# # full set: alpha+beta set + beta set (MVIB)
# In[17]:
#modality = "bimodal"
#joint_posterior = "poe"
#data_type = "full"
train_generic("bimodal", "poe", "full")

# # Max pooling of experts
# In[18]:
joint_posterior = "max_pool"
# # alpha+beta set - peptide+CDR3b (max pooling of experts)
# In[19]:
#modality = "bimodal"
#joint_posterior = "max_pool"
#data_type = "alpha+beta-only"
train_generic("bimodal", "max_pool", "alpha+beta-only")

# # alpha+beta set - peptide+CDR3b+CDR3a  (max pooling of experts)
# In[20]:
#modality = "trimodal"
#joint_posterior = "max_pool"
#data_type = "alpha+beta-only"
train_generic("trimodal", "max_pool", "alpha+beta-only")

# # Average pooling of experts
# In[21]:
joint_posterior = "avg_pool"
# # alpha+beta set - peptide+CDR3b (average pooling of experts)
# In[22]:
#modality = "bimodal"
#joint_posterior = "avg_pool"
#data_type = "alpha+beta-only"
train_generic("bimodal", "avg_pool", "alpha+beta-only")

# # alpha+beta set - peptide+CDR3b+CDR3a  (average pooling of experts)
# In[23]:
#modality = "trimodal"
#joint_posterior = "avg_pool"
#data_type = "alpha+beta-only"
train_generic("trimodal", "avg_pool", "alpha+beta-only")

# In[ ]:




