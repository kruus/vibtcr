#!/usr/bin/env python
# coding: utf-8

# # Baselines for TCR
# The notebook `dataset-creation.ipyn` creates two dataset: the `α+β set`, and the `β set`. The `α+β set` contains `(CDR3α, CDR3β, peptide)` samples. The `β set` contains `(CDR3β, peptide)` samples.
# 
# In this notebook, we do experiments on ERGO II, NetTCR2.0, and LUPI-SVM using them as baseline for our research. We train and test on the `α+β set`
# 
# For testing, we operate 5 independent train/test splits of `α+β set` with different random seeds.

# ## Utility functions

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import trange
import random
import math
from scipy import interp
import statistics 

from tcrmodels.ergo2.model import ERGO2
from tcrmodels.nettcr2.model import NetTCR2

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from matplotlib import collections
from matplotlib import colors
from numpy.random import normal


# In[2]:


metrics = [
    'AUROC',
    'Accuracy',
    #'Recall',
    'Precision',
    'F1 score',
    'AUPR'
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
        #recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        pr_auc(y_true, y_prob)
    ]
    
    df = pd.DataFrame(data={'score': scores, 'metrics': metrics})
    return df


# In[3]:


blosum50_20aa = {
    'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
    'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
    'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
    'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
    'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
    'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
    'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
    'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
    'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
    'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
    'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
    'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
    'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
    'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
    'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
    'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
    'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
    'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
    'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
    'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5))
}

def enc_list_bl_max_len(aas, blosum, max_seq_len):
    '''
    blosum encoding of a list of amino acid sequences with padding 
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        padded_aa_encoding : array of padded amino acids encoding
    '''
    encoding_len = len(blosum['A'])
    padded_aa_encoding = np.zeros((encoding_len * max_seq_len))
    
    # encode amino acids
    for i, aa in enumerate(aas):
        padded_aa_encoding[i*encoding_len:(i+1)*encoding_len] = blosum[aa]
        
    return padded_aa_encoding


# In[4]:


# Abbasi et al. implementation of LUPI-SVM does not predict probabilities
# we should do Platt scaling to estimate probabilities of a SVM
# but since that would be a complex implementation, we use a sigmoid
from scipy.stats import logistic

def sigmoid(x):
    return logistic.cdf(x)


# In[5]:


import os
login = os.getlogin( )
DATA_BASE = f"/home/{login}/Git/tcr/data/"
RESULTS_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/results/"
# To run in github checkout of vibtcr, after `unzip data.zip` ...
RESULTS_BASE = os.path.join('.', 'results')
FIGURES_BASE = os.path.join('.', 'figures')
DATA_BASE = os.path.join('..', '..', 'data')


# # NetTCR2.0 - Peptide+CDR3β

# In[6]:


get_ipython().run_cell_magic('capture', '', '\ndf_dataset = pd.read_csv(os.path.join(DATA_BASE, "alpha-beta-splits", "alpha-beta.csv"))\ntest_files = (\'(α+β-set(β-only)\', [])\nresults_nettcr2 = []\n\nfor i in tqdm(range(5)):\n    df_train, df_test = train_test_split(df_dataset, test_size=0.2, random_state=i)\n    test_files[1].append(df_test.reset_index())\n\n    model = NetTCR2(\n        architecture="b", \n        single_chain_column=\'tcrb\',\n        peptide_column=\'peptide\',\n        label_column=\'sign\',\n        max_pep_len=df_dataset.peptide.str.len().max(), \n        max_cdr3_len=df_dataset.tcrb.str.len().max()\n    )\n    model.train(df_train, epochs=1000);\n\n    prediction_df = model.test(test_files[1][i])\n    scores_df = get_scores(\n        y_true=prediction_df[\'sign\'].to_numpy(), \n        y_prob=prediction_df[\'prediction\'].to_numpy(),\n        y_pred=prediction_df[\'prediction\'].to_numpy().round(),\n    )\n    scores_df[\'experiment\'] = "train_α+β-set(β-only)_test_"+test_files[0]\n    results_nettcr2.append(scores_df)\n    test_files[1][i][\'prediction_\'+str(i)] = prediction_df[\'prediction\']\n        \nresults_nettcr2_df = pd.concat(results_nettcr2)\n\n# save results for further analysis\nfor i, test_file in enumerate(test_files[1]):\n    test_file.to_csv(os.path.join(RESULTS_BASE, f"nettcr2.baseline.alpha+beta-only.rep-{i}.csv"), index=False)\n')


# # NetTCR2.0 - Peptide+CDR3β+CDR3α

# In[7]:


get_ipython().run_cell_magic('capture', '', '\ndf_dataset = pd.read_csv(os.path.join(DATA_BASE, "alpha-beta-splits", "alpha-beta.csv"))\ntest_files = (\'(α+β-set(α+β)\', [])\nresults_nettcr2 = []\n\nfor i in tqdm(range(5)):\n    df_train, df_test = train_test_split(df_dataset, test_size=0.2, random_state=i)\n    test_files[1].append(df_test.reset_index())\n\n    model = NetTCR2(\n        architecture="ab", \n        cdr3b_column=\'tcrb\',\n        cdr3a_column=\'tcra\',\n        peptide_column=\'peptide\',\n        label_column=\'sign\',\n        max_pep_len=df_dataset.peptide.str.len().max(), \n        max_cdr3b_len=df_dataset.tcrb.str.len().max(),\n        max_cdr3a_len=df_dataset.tcra.str.len().max()\n    )\n    model.train(df_train, epochs=1000);\n\n    prediction_df = model.test(test_files[1][i])\n    scores_df = get_scores(\n        y_true=prediction_df[\'sign\'].to_numpy(), \n        y_prob=prediction_df[\'prediction\'].to_numpy(),\n        y_pred=prediction_df[\'prediction\'].to_numpy().round(),\n    )\n    scores_df[\'experiment\'] = "train_α+β-set(α+β)_test_"+test_files[0]\n    results_nettcr2.append(scores_df)\n    test_files[1][i][\'prediction_\'+str(i)] = prediction_df[\'prediction\']\n        \nresults_nettcr2_df = pd.concat(results_nettcr2)\n\n# save results for further analysis\nfor i, test_file in enumerate(test_files[1]):\n    test_file.to_csv(\n        os.path.join(RESULTS_BASE, f"nettcr2.baseline.alpha+beta-only.alpha+beta+peptide.rep-{i}.csv"),\n        index=False\n    )\n')


# # ERGO II - Peptide+CDR3β

# In[8]:


get_ipython().run_cell_magic('capture', '', '\n# the ERGO II data presents some files with a given header, and some others with a different one\nmap_keys = {\n    \'tcra\': \'TRA\',\n    \'tcrb\': \'TRB\',\n    \'va\': \'TRAV\',\n    \'ja\': \'TRAJ\',\n    \'vb\': \'TRBV\',\n    \'jb\': \'TRBJ\',\n    \'t_cell_type\': \'T-Cell-Type\',\n    \'peptide\': \'Peptide\',\n    \'mhc\': \'MHC\',\n    \'protein\': \'protein\',\n    \'sign\': \'sign\'\n}\n\ndf_dataset = pd.read_csv(os.path.join(DATA_BASE, "alpha-beta-splits", "alpha-beta.csv"))\nresults_ergo2 = []\ntest_files = (\'(α+β-set(β-only)\', [])\n\nfor i in tqdm(range(5)):\n    df_train, df_test = train_test_split(df_dataset, test_size=0.2, random_state=i)\n    \n    # the ERGO II implementation expected the following columns to be preset in the dataframe\n    # even if they are not used\n    df_test[\'va\'] = pd.NA\n    df_train[\'va\'] = pd.NA\n    df_test[\'vb\'] = pd.NA\n    df_train[\'vb\'] = pd.NA\n    df_test[\'ja\'] = pd.NA\n    df_train[\'ja\'] = pd.NA\n    df_test[\'jb\'] = pd.NA\n    df_train[\'jb\'] = pd.NA\n    df_test[\'mhc\'] = pd.NA\n    df_train[\'mhc\'] = pd.NA\n    df_test[\'t_cell_type\'] = pd.NA\n    df_train[\'t_cell_type\'] = pd.NA\n    df_test[\'protein\'] = pd.NA\n    df_train[\'protein\'] = pd.NA\n\n    # using "UNK" for identifier of missing CDR3α for test set\n    df_test[\'tcra\'] = "UNK"\n\n    df_test = df_test.rename(columns={c: map_keys[c] for c in df_test.columns})\n    \n    test_files[1].append(df_test.reset_index())\n\n    model = ERGO2(\n        gpu=[0],\n        use_alpha=False,\n        random_seed=i,\n        train_val_ratio=.2,\n    )\n    model.train(df_train);\n\n    prediction_df = model.test(test_files[1][i])\n    scores_df = get_scores(\n        y_true=prediction_df[\'sign\'].to_numpy(), \n        y_prob=prediction_df[\'prediction\'].to_numpy(),\n        y_pred=prediction_df[\'prediction\'].to_numpy().round(),\n    )\n    scores_df[\'experiment\'] = "train_α+β-set(β-only)_test_"+test_files[0]\n    results_ergo2.append(scores_df)\n    test_files[1][i][\'prediction_\'+str(i)] = prediction_df[\'prediction\']\n        \nresults_ergo2_df = pd.concat(results_ergo2)\n\n# save results for further analysis\nfor i, test_file in enumerate(test_files[1]):\n    test_file.to_csv(os.path.join(RESULTS_BASE, f"ergo2.baseline.alpha+beta-only.rep-{i}.csv"), index=False)\n')


# # ERGO II - Peptide+CDR3β+CDR3α

# In[9]:


get_ipython().run_cell_magic('capture', '', '\n# the ERGO II data presents some files with a given header, and some others with a different one\nmap_keys = {\n    \'tcra\': \'TRA\',\n    \'tcrb\': \'TRB\',\n    \'va\': \'TRAV\',\n    \'ja\': \'TRAJ\',\n    \'vb\': \'TRBV\',\n    \'jb\': \'TRBJ\',\n    \'t_cell_type\': \'T-Cell-Type\',\n    \'peptide\': \'Peptide\',\n    \'mhc\': \'MHC\',\n    \'protein\': \'protein\',\n    \'sign\': \'sign\'\n}\n\ndf_dataset = pd.read_csv(os.path.join(DATA_BASE, "alpha-beta-splits", "alpha-beta.csv"))\nresults_ergo2 = []\ntest_files = (\'(α+β-set(α+β)\', [])\n\nfor i in tqdm(range(5)):\n    df_train, df_test = train_test_split(df_dataset, test_size=0.2, random_state=i)\n    \n    # the ERGO II implementation expected the following columns to be preset in the dataframe\n    # even if they are not used\n    df_test[\'va\'] = pd.NA\n    df_train[\'va\'] = pd.NA\n    df_test[\'vb\'] = pd.NA\n    df_train[\'vb\'] = pd.NA\n    df_test[\'ja\'] = pd.NA\n    df_train[\'ja\'] = pd.NA\n    df_test[\'jb\'] = pd.NA\n    df_train[\'jb\'] = pd.NA\n    df_test[\'mhc\'] = pd.NA\n    df_train[\'mhc\'] = pd.NA\n    df_test[\'t_cell_type\'] = pd.NA\n    df_train[\'t_cell_type\'] = pd.NA\n    df_test[\'protein\'] = pd.NA\n    df_train[\'protein\'] = pd.NA\n\n    df_test = df_test.rename(columns={c: map_keys[c] for c in df_test.columns})\n    \n    test_files[1].append(df_test.reset_index())\n\n    model = ERGO2(\n        gpu=[0],\n        use_alpha=True,\n        random_seed=i,\n        train_val_ratio=.2,\n    )\n    model.train(df_train);\n\n    prediction_df = model.test(test_files[1][i])\n    scores_df = get_scores(\n        y_true=prediction_df[\'sign\'].to_numpy(), \n        y_prob=prediction_df[\'prediction\'].to_numpy(),\n        y_pred=prediction_df[\'prediction\'].to_numpy().round(),\n    )\n    scores_df[\'experiment\'] = "train_α+β-set(α+β)_test_"+test_files[0]\n    results_ergo2.append(scores_df)\n    test_files[1][i][\'prediction_\'+str(i)] = prediction_df[\'prediction\']\n        \nresults_ergo2_df = pd.concat(results_ergo2)\n\n# save results for further analysis\nfor i, test_file in enumerate(test_files[1]):\n    test_file.to_csv(os.path.join(RESULTS_BASE, f"ergo2.baseline.alpha+beta-only.alpha+beta+peptide.rep-{i}.csv"), index=False)\n')


# # LUPI-SVM - CDR3α-privileged
# Learning protein binding affinity using privileged information, Abbasi et al., BMC Bioinformatics, 2018
# 
# Paper: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2448-z
# 
# Code: https://github.com/wajidarshad/LUPI-SVM

# In[10]:


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 00:59:45 2017

@author: Wajid Arshad Abbasi

This module contains the class definitions for the Stochastic subgradient descent based large margin classifiers for Learning Using Privileged Information (LUPI)

"""
class ClassifierBase:
    """
    This is the base class for LUPI
    """
    
    def __init__(self,**kwargs):
    
        if 'epochs' in kwargs:
            self.epochs=kwargs['epochs']
        else:
            self.epochs=100
        if 'Lambda' in kwargs:
            self.Lambda=kwargs['Lambda']
        else:
            self.Lambda=0.01
        if 'Lambda_star' in kwargs:
            self.Lambda_star=kwargs['Lambda_star']
        else:
            self.Lambda_star=0.01
        if 'Lambda_s' in kwargs:
            self.Lambda_s=kwargs['Lambda_s']
        else:
            self.Lambda_s=0.001
        self.w=None
        self.w_star=None
        self.Name=None
        
    def fit(self,bags,**kwargs):
        pass
        
        
    def predict_score(self,test_example):
        w=self.w
        pred_score=test_example.dot(w.T)
        return pred_score
    def save(self,ofname):
        with open(ofname,'w') as fout:
            fout.write(self.toString())
    def load(self,ifname):
        with open(ifname) as fin:
           self.fromString(fin.read())         
    def toString(self):
        import json
        s='#Name='+str(self.__class__)
        s+='#w='+str(json.dumps(self.w.tolist()))
        s+='#w_star='+str(json.dumps(self.w_star.tolist()))
        s+='#Epochs='+str(self.epochs)  
        s+='#Lambda='+str(self.Lambda)
        s+='#Lambda_star='+str(self.Lambda_star)
        s+='#Lambda_s='+str(self.Lambda_s)
        return s
        
    def fromString(self,s):    
        import json
        for token in s.split('#'):
            if token.find('w=')>=0 or token.find('W=')>=0:
                self.w=np.array(json.loads(token.split('=')[1]))
            if token.find('w_star=')>=0 or token.find('W_star=')>=0:
                self.w_star=np.array(json.loads(token.split('=')[1]))
            elif token.find('Epochs=')>=0:
                self.epochs=float(token.split('=')[1]) 
            elif token.find('Lambda_star=')>=0:
                self.Lambda_star=float(token.split('=')[1])
            elif token.find('Lambda=')>=0:
                self.Lambda=float(token.split('=')[1])
            elif token.find('Lambda_s=')>=0:
                self.Lambda_s=float(token.split('=')[1])

#############################################################################################

class linclassLUPI(ClassifierBase):   
    """
    This class defines the stochastic gradient descent based linear large margin classifier for LUPI.

    Parent Class: ClassifierBase
    
    Properties:
    epochs: No. of epochs to be run for optimization
    Lambda, Lambda_satr and Lambda_s: The Regularization Hyperparameters
    
    Methods:
    train(dataset)
    predict(example)
    load(filename)
    save(filename)
    
    USAGE
    Class definition:
    clf=linclassLUPI() # create a classifier object with default arguments epochs=100, Lambda=0.01, Lambda_star=0.01, Lambda_s=0.001
    clf=linclassLUPI(epochs=100, Lambda=0.01,Lambda_star=0.1,Lambda_s=0.001) # create a classifier object with customized arguments
    
    Training:
    clf.fit(clf.train([[[x1],[X1*],y1],[x2],[X2*],y2],[x3],[X3*],y3],....[Xn],[Xn*],yn]])) where X:Input Feature Space, X*: Privileged Feature Space and y: Labels
    
    Predict:
    clf.predict_score([[X_test1],[X_test2]]) X_test: test examples only input feature space
    
    Load Classifier:
    clf.load(filename)
    
    Save Classifier:
    clf.save(filename)
    """
    
    def fit(self, dataset,**kwargs):
        
        siz1=np.shape(dataset[0][0])[0]
        siz2=np.shape(dataset[0][1])[0]
        w=np.array(np.zeros(siz1))  
        w_star=np.array(np.zeros(siz2))
        T=(len(dataset))*self.epochs
        for t in trange(T):
            mue=1.0/(self.Lambda*(t+1))
            mue_star=1.0/(self.Lambda_star*(t+1))
            update_w=False
            update_w_star=False
            if (t)%self.epochs==0:
                np.random.shuffle(dataset)
            instance_chosen=dataset[(t-1)%len(dataset)]
            if 1-instance_chosen[2]*(instance_chosen[0].dot(w.T))-instance_chosen[2]*(instance_chosen[1].dot(w_star.T))>0 and 1-instance_chosen[2]*(instance_chosen[0].dot(w.T))>0:
                update_w=True
            if -instance_chosen[2]*(instance_chosen[1].dot(w_star.T))>0 or 1-instance_chosen[2]*(instance_chosen[0].dot(w.T))-instance_chosen[2]*(instance_chosen[1].dot(w_star.T))>0:
                update_w_star=True
            if update_w:
                w=((1-(1.0/(t+1)))*w)+(mue*(instance_chosen[2]*instance_chosen[0]))
            else:
                w=((1-(1.0/(t+1)))*w)
            if update_w_star:
                w_star=((1-(1.0/(t+1)))*w_star)-(mue_star*self.Lambda_s*(instance_chosen[2]*instance_chosen[1]))+(mue_star*(instance_chosen[2]*instance_chosen[1]))
            else:
                w_star=((1-(1.0/(t+1)))*w_star)-(mue_star*self.Lambda_s*(instance_chosen[2]*instance_chosen[1]))
        self.w=w
        self.w_star=w_star

#####################################################################################################################################


# In[11]:


df_dataset = pd.read_csv(os.path.join(DATA_BASE, "alpha-beta-splits", "alpha-beta.csv"))
test_files = ('(α+β-set(β-only)', [])

for i in range(5):
    df_train, df_test = train_test_split(df_dataset, test_size=0.2, random_state=i)

    test_files[1].append(df_test.reset_index())
    
    # create X_train, X_test, y_train, y_test
    max_tcra_len = df_train.tcra.str.len().max()
    df_train['tcra'] = df_train['tcra'].apply(lambda x: enc_list_bl_max_len(x, blosum50_20aa, max_tcra_len))

    max_tcrb_len = max(df_train.tcrb.str.len().max(), test_files[1][i].tcrb.str.len().max())
    df_train['tcrb'] = df_train['tcrb'].apply(lambda x: enc_list_bl_max_len(x, blosum50_20aa, max_tcrb_len))
    test_files[1][i]['tcrb'] = test_files[1][i]['tcrb'].apply(lambda x: enc_list_bl_max_len(x, blosum50_20aa, max_tcrb_len))

    max_pep_len = max(df_train.peptide.str.len().max(), test_files[1][i].peptide.str.len().max())
    df_train['peptide'] = df_train['peptide'].apply(lambda x: enc_list_bl_max_len(x, blosum50_20aa, max_pep_len))
    test_files[1][i]['peptide'] = test_files[1][i]['peptide'].apply(lambda x: enc_list_bl_max_len(x, blosum50_20aa, max_pep_len))
    
    X_privileged = np.stack(df_train.tcra.to_list(), axis=0)

    tcrb = np.stack(df_train.tcrb.to_list(), axis=0)
    peptides = np.stack(df_train.peptide.to_list(), axis=0)
    X_train = np.concatenate([tcrb, peptides], axis=1)

    tcrb = np.stack(test_files[1][i].tcrb.to_list(), axis=0)
    peptides = np.stack(test_files[1][i].peptide.to_list(), axis=0)
    X_test = np.concatenate([tcrb, peptides], axis=1)

    y_train = np.stack(df_train.sign.to_list(), axis=0)

    y_test = np.stack(test_files[1][i].sign.to_list(), axis=0)
    
    # scale the dataset
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_privileged = scaler.fit_transform(X_privileged)
    
    # train LUPI-SVM
    clf=linclassLUPI(epochs=1000, Lambda=0.1,Lambda_star=0.01,Lambda_s=0.001)
    LUPI_train_data = list(zip(X_train, X_privileged, y_train))
    clf.fit(LUPI_train_data)
    
    # test
    y_decision_function = clf.predict_score(X_test)
    y_prob = sigmoid(y_decision_function)  # should be Platt scaling
    y_pred = np.sign(y_decision_function).clip(min=0)
    scores_df = get_scores(y_test, y_prob, y_pred)
    test_files[1][i]['prediction_'+str(i)] = list(y_decision_function)

# save results for further analysis
for i, test_file in enumerate(test_files[1]):
    test_file.to_csv(os.path.join(RESULTS_BASE, f"lupi-svm.baseline.alpha+beta-only.rep-{i}.csv"), index=False)


# # Figures

# In[12]:


import os
login = os.getlogin( )
DATA_BASE = f"/home/{login}/Git/tcr/data/"
RESULTS_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/results/"
FIGURES_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/figures/"
# To run in github checkout of vibtcr, after `unzip data.zip` ...
RESULTS_BASE = os.path.join('.', 'results')
FIGURES_BASE = os.path.join('.', 'figures')
DATA_BASE = os.path.join('..', '..', 'data')
predictions_files = [
    ('LUPI-SVM | α-privileged', [pd.read_csv(os.path.join(RESULTS_BASE, f"lupi-svm.baseline.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('NetTCR2.0 | peptide+β',   [pd.read_csv(os.path.join(RESULTS_BASE, f"nettcr2.baseline.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('NetTCR2.0 | peptide+α+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"nettcr2.baseline.alpha+beta-only.alpha+beta+peptide.rep-{i}.csv")) for i in range(5)]),
    ('ERGO II | peptide+β',     [pd.read_csv(os.path.join(RESULTS_BASE, f"ergo2.baseline.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('ERGO II | peptide+α+β',   [pd.read_csv(os.path.join(RESULTS_BASE, f"ergo2.baseline.alpha+beta-only.alpha+beta+peptide.rep-{i}.csv")) for i in range(5)]),
    ('AVIB | peptide+β',        [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.aoe.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('AVIB | peptide+α+β',      [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.trimodal.aoe.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('AVIB | peptide+α',        [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal-alpha.aoe.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
]


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('axes', axisbelow=True)

results = []

for i in tqdm(range(5)):
    for predictions_file in predictions_files:
        prediction_df = predictions_file[1][i]
        if f'prediction_{i}' in prediction_df.columns:
            if "LUPI-SVM" in predictions_file[0]:
                scores_df = get_scores(
                    y_true=prediction_df['sign'].to_numpy(), 
                    y_prob=sigmoid(prediction_df[f'prediction_{i}'].to_numpy()),
                    y_pred=np.sign(prediction_df[f'prediction_{i}'].to_numpy().round()).clip(min=0),
                )
            else:
                scores_df = get_scores(
                    y_true=prediction_df['sign'].to_numpy(), 
                    y_prob=prediction_df[f'prediction_{i}'].to_numpy(),
                    y_pred=prediction_df[f'prediction_{i}'].to_numpy().round(),
                )
            scores_df['Model'] = predictions_file[0]
            results.append(scores_df)
        
results_df = pd.concat(results).rename(columns={'metrics': 'Metrics', 'score': 'Score'})

plt.rcParams['figure.figsize'] = [10, 6]
ax = sns.barplot(
    x="Metrics",
    y="Score", 
    hue="Model", 
    data=results_df,
    palette=sns.color_palette("magma", len(predictions_files))
)
ax.set_title('TCR recognition | Train: α+β-set | Test: α+β-set')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='best')
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
ax.grid(axis='y')

plt.savefig(os.path.join(FIGURES_BASE, "baseline.alpha-beta-only.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "baseline.alpha-beta-only.png"), format='png', dpi=300, bbox_inches='tight')


# In[14]:


results_df.groupby(['Metrics', 'Model']).mean()


# In[15]:


ste = results_df.groupby(['Metrics', 'Model']).std()
ste['Score'] = ste['Score'].apply(lambda x: x / 5)
ste


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('axes', axisbelow=True)

plt.style.use('seaborn-white')
sns.set_palette('magma', len(predictions_files))


def make_roc_curve_plot(ax, true_values_list, predicted_values_list, cutoff, model_label):
    """Calculate ROC and AUC from lists of true and predicted values and draw."""

    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    auc = []
    for true_values, predicted_values in zip(true_values_list, predicted_values_list):
        fpr, tpr, thresholds = roc_curve(true_values, predicted_values)
        auc.append(roc_auc_score(true_values, predicted_values))
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)

    ax.plot(base_fpr, mean_tprs, label=model_label+str(f" | AUROC: {statistics.mean(auc):.3f}"))
    
    ax.set_title("ROC Curve | Train: α+β-set | Test: α+β-set")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
#     for fp, tp, threshold in zip(fpr, tpr, thresholds):
#         if threshold < cutoff:
#             ax.plot(fp, tp, marker='o', markersize=10, color='grey', alpha=0.75)
#             break


def make_uninformative_roc(ax):
    ax.plot([0, 1], [0, 1], c='grey', linestyle='dashed', alpha=0.5, label="Uninformative test")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(facecolor="white")


fig, ax = plt.subplots()

for predictions_file in predictions_files:
    true_values_list, predicted_values_list = [], []
    for i in range(5):
        prediction_df = predictions_file[1][i]
        true_values_list.append(prediction_df['sign'].to_numpy())
        if "LUPI-SVM" in predictions_file[0]:
            predicted_values_list.append(sigmoid(prediction_df[f'prediction_{i}'].to_numpy()))
        else:
            predicted_values_list.append(prediction_df[f'prediction_{i}'].to_numpy())

    make_roc_curve_plot(
        ax, 
        true_values_list, 
        predicted_values_list, 
        0.9,
        predictions_file[0]
    )
make_uninformative_roc(ax)
ax.tick_params(axis='x', pad=15)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='best')
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')

plt.savefig(os.path.join(FIGURES_BASE, "roc.alpha-beta-only.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "roc.alpha-beta-only.png"), format='png', dpi=300, bbox_inches='tight')


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

def make_prc_curve_plot(ax, true_values, predicted_values, model_label):
    """Calculate PRC and AUC from lists of true and predicted values and draw."""
    
    reversed_mean_precision = 0.0
    base_recall = np.linspace(1, 0, 100)
    auc = []
    
    for true_values, predicted_values in zip(true_values_list, predicted_values_list):
        precision, recall, thresholds = precision_recall_curve(true_values, predicted_values)
        auc.append(pr_auc(true_values, predicted_values))
        reversed_recall = np.fliplr([recall])[0]
        reversed_precision = np.fliplr([precision])[0]
        reversed_mean_precision += interp(base_recall, reversed_recall, reversed_precision)
    
    reversed_mean_precision /= 5
    
    ax.plot(base_recall, reversed_mean_precision, label=model_label+str(f" | AUPR: {statistics.mean(auc):.3f}"))
    
    ax.set_title("Precision-Recall Curve | Train: α+β-set | Test: α+β-set")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.legend()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])


fig, ax = plt.subplots()

for predictions_file in predictions_files:
    true_values_list, predicted_values_list = [], []
    for i in range(5):
        prediction_df = predictions_file[1][i]
        true_values_list.append(prediction_df['sign'].to_numpy())
        if "LUPI-SVM" in predictions_file[0]:  # LUPI-SSVM
            predicted_values_list.append(sigmoid(prediction_df[f'prediction_{i}'].to_numpy()))
        else:
            predicted_values_list.append(prediction_df[f'prediction_{i}'].to_numpy())

    make_prc_curve_plot(
        ax, 
        true_values_list, 
        predicted_values_list, 
        predictions_file[0]
    )

ax.tick_params(axis='x', pad=15)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='best')
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
plt.savefig(os.path.join(FIGURES_BASE, "prc.alpha-beta-only.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "prc.alpha-beta-only.png"), format='png', dpi=300, bbox_inches='tight')


# # MVIB - Experts comparison

# In[18]:


import os
login = os.getlogin( )
DATA_BASE = f"/home/{login}/Git/tcr/data/"
RESULTS_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/results/"
FIGURES_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/figures/"
# To run in github checkout of vibtcr, after `unzip data.zip` ...
RESULTS_BASE = os.path.join('.', 'results')
FIGURES_BASE = os.path.join('.', 'figures')
DATA_BASE = os.path.join('..', '..', 'data')
predictions_files = [
    ('AVIB (AoE) | peptide+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.aoe.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('MVIB (PoE) | peptide+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.poe.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('MaxPoE | peptide+β',     [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.max_pool.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('AvgPoE | peptide+β',     [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.avg_pool.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),

    ('AVIB (AoE) | peptide+α+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.trimodal.aoe.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('MVIB (PoE) | peptide+α+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.trimodal.poe.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('MaxPoE | peptide+α+β',     [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.trimodal.max_pool.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
    ('AvgPoE | peptide+α+β',     [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.trimodal.avg_pool.alpha+beta-only.rep-{i}.csv")) for i in range(5)]),
]


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt


results = []
colors = ["#BAF1FF",  "#FDFD98", "#55DCFF", "#F5F542", ]

for i in tqdm(range(5)):
    for predictions_file in predictions_files:
        prediction_df = predictions_file[1][i]
        if f'prediction_{i}' in prediction_df.columns:
            scores_df = get_scores(
                y_true=prediction_df['sign'].to_numpy(), 
                y_prob=prediction_df[f'prediction_{i}'].to_numpy(),
                y_pred=prediction_df[f'prediction_{i}'].to_numpy().round(),
            )
            scores_df['Model'] = predictions_file[0]
            results.append(scores_df)
        
results_df = pd.concat(results).rename(columns={'metrics': 'Metrics', 'score': 'Score'})

plt.rcParams['figure.figsize'] = [10, 6]
ax = sns.barplot(
    x="Metrics",
    y="Score", 
    hue="Model", 
    data=results_df,
    palette=sns.color_palette('magma', 4)
)
ax.set_title('Experts comparison | Train: α+β-set | Test: α+β-set')
ax.legend(loc='best')
ax.grid(axis='y')
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
plt.savefig(os.path.join(FIGURES_BASE, "experts-comparison.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "experts-comparison.png"), format='png', dpi=300, bbox_inches='tight')


# In[20]:


results_df.groupby(['Metrics', 'Model']).mean()


# In[21]:


std_df = results_df.groupby(['Metrics', 'Model']).std()
std_df['Score'] = std_df['Score'].apply(lambda x: x / 5)
std_df


# In[ ]:




