#!/usr/bin/env python
# coding: utf-8

# # Attentive Variational Information Bottleneck
# 
# In this notebook, we train the Attentive Variational Information Bottleneck on the `α+β set` and test on the `β set`.

# In[1]:


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


# In[2]:


import os
login = os.getlogin( )
DATA_BASE = f"/home/{login}/Git/tcr/data/"
RESULTS_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/results/"
# To run in github checkout of vibtcr, after `unzip data.zip` ...
RESULTS_BASE = os.path.join('.', 'results')
#FIGURES_BASE = os.path.join('.', 'figures')
DATA_BASE = os.path.join('..', '..', 'data')


# In[3]:


device = torch.device('cuda:1')
batch_size = 4096
epochs = 500 #200
lr = 1e-3
z_dim = 150
beta = 1e-6
early_stopper_patience = 20
monitor = 'auROC'
lr_scheduler_param = 10
joint_posterior = "aoe"


# In[4]:


import pandas as pd
import torch
import numpy as np

from vibtcr.dataset import TCRDataset
from vibtcr.mvib.mvib import MVIB
from vibtcr.mvib.mvib_trainer import TrainerMVIB
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


df = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'alpha-beta.csv'))
scaler = TCRDataset(df.copy(), torch.device("cpu"), cdr3b_col='tcrb', cdr3a_col=None).scaler

df_test = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'beta.csv'))
ds_test = TCRDataset(df_test, torch.device("cpu"), cdr3b_col='tcrb', cdr3a_col=None, scaler=scaler)

for i in range(5):  # 5 independent train/val splits
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df.sign, random_state=i)
    
    # train loader with balanced sampling
    ds_train = TCRDataset(df_train, device, cdr3b_col='tcrb', cdr3a_col=None, scaler=scaler)
    class_count = np.array([df_train[df_train.sign == 0].shape[0], df_train[df_train.sign == 1].shape[0]])
    weight = 1. / class_count
    samples_weight = torch.tensor([weight[s] for s in df_train.sign])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=sampler
    )
    
    # val loader with balanced sampling
    ds_val = TCRDataset(df_val, device, cdr3b_col='tcrb', cdr3a_col=None, scaler=scaler)
    class_count = np.array([df_val[df_val.sign == 0].shape[0], df_val[df_val.sign == 1].shape[0]])
    weight = 1. / class_count
    samples_weight = torch.tensor([weight[s] for s in df_val.sign])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    val_loader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=batch_size,
        sampler=sampler
    )

    model = MVIB(z_dim=z_dim, device=device, joint_posterior=joint_posterior).to(device)

    trainer = TrainerMVIB(
        model,
        epochs=epochs,
        lr=lr,
        beta=beta,
        checkpoint_dir=".",
        mode="bimodal",
        lr_scheduler_param=lr_scheduler_param
    )
    checkpoint = trainer.train(train_loader, val_loader, early_stopper_patience, monitor)
    run_name = f"mvib.ab2b-rep{i}"
    trainer.save_checkpoint(checkpoint, folder='./', filename=os.path.join(RESULTS_BASE, f"{run_name}.pth"))
    
    # test
    model = MVIB.from_checkpoint(checkpoint, torch.device("cpu"))
    pred = model.classify(pep=ds_test.pep, cdr3b=ds_test.cdr3b, cdr3a=None)
    pred = pred.detach().numpy()
    df_test['prediction_'+str(i)] = pred.squeeze().tolist()

# save results for further analysis
df_test.to_csv(os.path.join(RESULTS_BASE, "mvib.ab2b.csv"), index=False)

