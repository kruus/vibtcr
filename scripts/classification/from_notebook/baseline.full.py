#!/usr/bin/env python
# coding: utf-8

# # Baselines for TCR
# The notebook `dataset-creation.ipyn` creates two dataset: the `α+β set`, and the `β set`. The `α+β set` contains `(CDR3α, CDR3β, peptide)` samples. The `β set` contains `(CDR3β, peptide)` samples.
# 
# In this notebook, we do experiments on ERGO II and NetTCR2.0 using them as baseline for our research. We train and test on the full set: `α+β set` + `β set`.
# 
# For testing, we operate 5 independent train/test splits of the full set with different random seeds.

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
FIGURES_BASE = f"/home/{login}/Git/tcr/notebooks/notebooks.classification/figures/"
# To run in github checkout of vibtcr, after `unzip data.zip` ...
DATA_BASE = os.path.join('..', '..', 'data')
RESULTS_BASE = os.path.join('.', 'results')
FIGURES_BASE = os.path.join('.', 'figures')


# # NetTCR2.0 - Peptide+CDR3β

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\ndf1 = pd.read_csv(os.path.join(DATA_BASE, \'alpha-beta-splits", "beta.csv\'))\ndf2 = pd.read_csv(os.path.join(DATA_BASE, \'alpha-beta-splits", "alpha-beta.csv\'))\ndf_dataset = pd.concat([df1, df2]).reset_index()\n\ntest_files = (\'full\', [])\n\nresults_nettcr2 = []\n\nfor i in tqdm(range(5)):\n    df_train, df_test = train_test_split(df_dataset, test_size=0.2, random_state=i)\n    test_files[1].append(df_test.reset_index())\n\n    model = NetTCR2(\n        architecture="b", \n        single_chain_column=\'tcrb\',\n        peptide_column=\'peptide\',\n        label_column=\'sign\',\n        max_pep_len=df_dataset.peptide.str.len().max(), \n        max_cdr3_len=df_dataset.tcrb.str.len().max()\n    )\n    model.train(df_train, epochs=1000);\n\n    prediction_df = model.test(test_files[1][i])\n    scores_df = get_scores(\n        y_true=prediction_df[\'sign\'].to_numpy(), \n        y_prob=prediction_df[\'prediction\'].to_numpy(),\n        y_pred=prediction_df[\'prediction\'].to_numpy().round(),\n    )\n    scores_df[\'experiment\'] = "train_full_test_"+test_files[0]\n    results_nettcr2.append(scores_df)\n    test_files[1][i][\'prediction_\'+str(i)] = prediction_df[\'prediction\']\n        \nresults_nettcr2_df = pd.concat(results_nettcr2)\n\n# save results for further analysis\nfor i, test_file in enumerate(test_files[1]):\n    test_file.to_csv(\n        os.path.join(RESULTS_BASE, f"nettcr2.baseline.full.rep-{i}.csv"),\n        index=False\n    )\n')


# # ERGO II - Peptide+CDR3β

# In[9]:


get_ipython().run_cell_magic('capture', '', '\n# the ERGO II data presents some files with a given header, and some others with a different one\nmap_keys = {\n    \'tcra\': \'TRA\',\n    \'tcrb\': \'TRB\',\n    \'va\': \'TRAV\',\n    \'ja\': \'TRAJ\',\n    \'vb\': \'TRBV\',\n    \'jb\': \'TRBJ\',\n    \'t_cell_type\': \'T-Cell-Type\',\n    \'peptide\': \'Peptide\',\n    \'mhc\': \'MHC\',\n    \'protein\': \'protein\',\n    \'sign\': \'sign\'\n}\n\ndf1 = pd.read_csv(os.path.join(DATA_BASE, \'alpha-beta-splits\', \'beta.csv\'))\ndf2 = pd.read_csv(os.path.join(DATA_BASE, \'alpha-beta-splits\', \'alpha-beta.csv\'))\ndf_dataset = pd.concat([df1, df2]).reset_index(drop=True)\n\ntest_files = (\'full\', [])\nresults_ergo2 = []\n\nfor i in tqdm(range(5)):\n    df_train, df_test = train_test_split(df_dataset, test_size=0.2, random_state=i)\n    \n    # the ERGO II implementation expected the following columns to be preset in the dataframe\n    # even if they are not used\n    df_test[\'va\'] = pd.NA\n    df_train[\'va\'] = pd.NA\n    df_test[\'vb\'] = pd.NA\n    df_train[\'vb\'] = pd.NA\n    df_test[\'ja\'] = pd.NA\n    df_train[\'ja\'] = pd.NA\n    df_test[\'jb\'] = pd.NA\n    df_train[\'jb\'] = pd.NA\n    df_test[\'mhc\'] = pd.NA\n    df_train[\'mhc\'] = pd.NA\n    df_test[\'t_cell_type\'] = pd.NA\n    df_train[\'t_cell_type\'] = pd.NA\n    df_test[\'protein\'] = pd.NA\n    df_train[\'protein\'] = pd.NA\n\n    # using "UNK" for identifier of missing CDR3α for test set\n    df_test[\'tcra\'] = "UNK"\n    df_train[\'tcra\'] = "UNK"\n\n    df_test = df_test.rename(columns={c: map_keys[c] for c in df_test.columns})\n    \n    test_files[1].append(df_test.reset_index())\n\n    model = ERGO2(\n        gpu=[0],\n        use_alpha=False,\n        random_seed=i,\n        train_val_ratio=.2,\n    )\n    model.train(df_train);\n\n    prediction_df = model.test(test_files[1][i])\n    scores_df = get_scores(\n        y_true=prediction_df[\'sign\'].to_numpy(), \n        y_prob=prediction_df[\'prediction\'].to_numpy(),\n        y_pred=prediction_df[\'prediction\'].to_numpy().round(),\n    )\n    scores_df[\'experiment\'] = "train_full-set_test_"+test_files[0]\n    results_ergo2.append(scores_df)\n    test_files[1][i][\'prediction_\'+str(i)] = prediction_df[\'prediction\']\n        \nresults_ergo2_df = pd.concat(results_ergo2)\n\n# save results for further analysis\nfor i, test_file in enumerate(test_files[1]):\n    test_file.to_csv(os.path.join(RESULTS_BASE, f"ergo2.baseline.full.rep-{i}.csv"), index=False)\n')


# # Figures

# In[6]:


predictions_files = [
    ('NetTCR2.0 | peptide+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"nettcr2.baseline.full.rep-{i}.csv")) for i in range(5)]),
    ('ERGO II | peptide+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"ergo2.baseline.full.rep-{i}.csv")) for i in range(5)]),
    ('AVIB | peptide+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.aoe.full.rep-{i}.csv")) for i in range(5)]),
]


# In[7]:


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
ax.set_title('TCR recognition | Train: full human set | Test: full human set')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='best')
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
ax.grid(axis='y')

plt.savefig(os.path.join(FIGURES_BASE, "baseline.full.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "baseline.full.png"), format='png', dpi=300, bbox_inches='tight')


# In[ ]:


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
    
    ax.set_title("ROC Curve | Train: full human set | Test: full human set")
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

plt.savefig(os.path.join(FIGURES_BASE, "roc.full.svg)", format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "roc.full.png)", format='png', dpi=300, bbox_inches='tight')


# In[8]:


results_df.groupby(['Metrics', 'Model']).mean()


# In[9]:


ste = results_df.groupby(['Metrics', 'Model']).std()
ste['Score'] = ste['Score'].apply(lambda x: x / 5)
ste


# In[ ]:


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
    
    ax.set_title("Precision-Recall Curve | Train: full human set | Test: full human set")
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
plt.savefig(os.path.join(FIGURES_BASE, "prc.full.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "prc.full.png"), format='png', dpi=300, bbox_inches='tight')


# # MVIB - Experts comparison

# In[20]:


predictions_files = [
    ('AVIB (AoE) | peptide+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.aoe.full.rep-{i}.csv")) for i in range(5)]),
    ('MVIB (PoE) | peptide+β', [pd.read_csv(os.path.join(RESULTS_BASE, f"mvib.bimodal.poe.full.rep-{i}.csv")) for i in range(5)]),
]


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('axes', axisbelow=True)


results = []
colors = ["#55DCFF", "#F5F542", ]

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
    palette=sns.color_palette(colors, 4)
)
ax.set_title('Experts comparison | Train: human set | Test: human set')
ax.legend(loc='best')
ax.grid(axis='y')
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
plt.savefig(os.path.join(FIGURES_BASE, "experts-comparison.full.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_BASE, "experts-comparison.full.png"), format='png', dpi=300, bbox_inches='tight')


# In[22]:


results_df.groupby(['Metrics', 'Model']).mean()


# In[23]:


results_df.groupby(['Metrics', 'Model']).std()


# In[ ]:




