#!/usr/bin/env python
# coding: utf-8

# # Attentive Variational Information Bottleneck
# 
# In this notebook, we train and test the Attentive Variational Information Bottleneck (MVIB [1] with Attention of Experts) and MVIB on all datasets.
# 
# [1] Microbiome-based disease prediction with multimodal variational information bottlenecks, Grazioli et al., https://www.biorxiv.org/node/2109522.external-links.html

# In[1]:

# I have 2 gpus.  split up avib.py into two sections
# SPLIT the training across 2 gpus (hacky, could use some multiprocessing to set up 2 threads, but...)
#
# NOTE: This notebook runs several choices for "joint_posterior"
#       Class BaseVIB supports:
#         'peo' ~ "product of experts" ~ "MVIB" (multimodal) original paper
#         'aoe' ~ "attention of experts" ~ "AVIB" (attentive)
#         'max_pool'
#         'avg_pool'
#joint_posterior = "aoe"

from avib_train_generic.py import train_generic

device = 'cuda'
train_generic("bimodal", "aoe", "alpha+beta-only", device=device)
train_generic("trimodal", "aoe", "alpha+beta-only", device=device)
train_generic("bimodal-alpha", "aoe", "alpha+beta-only", device=device)
train_generic("bimodal", "aoe", "beta-only", device=device)
train_generic("bimodal", "aoe", "full", device=device)
train_generic("bimodal", "poe", "alpha+beta-only", device=device)
train_generic("trimodal", "poe", "alpha+beta-only", device=device)
train_generic("bimodal-alpha", "poe", "alpha+beta-only", device=device)
train_generic("bimodal", "poe", "beta-only", device=device)
train_generic("bimodal", "poe", "full", device=device)
train_generic("bimodal", "max_pool", "alpha+beta-only", device=device)
train_generic("trimodal", "max_pool", "alpha+beta-only", device=device)
train_generic("bimodal", "avg_pool", "alpha+beta-only", device=device)
train_generic("trimodal", "avg_pool", "alpha+beta-only", device=device)

# In[ ]:




