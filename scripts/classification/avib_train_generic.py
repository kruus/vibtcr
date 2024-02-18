#device = torch.device('cuda')

import pandas as pd
import torch
import numpy as np
import random
from pathlib import Path

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

import contextlib

def train_generic(
    modality:str,
    joint_posterior:str,
    data_type:str,
    device:str = "cuda",
    folds:int = 5,
    recalculate:bool = False,
    dry_run:bool = False
):
    assert modality        in ['bimodal', 'trimodal', 'bimodal-alpha']
    assert joint_posterior in ['aoe', 'poe', 'max_pool', 'avg_pool']
    assert data_type       in ["alpha+beta-only", "full", "beta-only"]
    run_name_base = f"mvib.{modality}.{joint_posterior}.{data_type}" #.rep-{i}
    print(f"\ntrain_generic {run_name_base}..."
          f"\ntrain_generic({modality},{joint_posterior},{data_type},{device=},{folds=},{recalculate=},{dry_run=}) Data...",flush=True)
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

    #if dry_run:
    #    return

    #with open(f'{run_name_base}.log', 'w') as f:
    #    with contextlib.redirect_stdout(f):
    if True:
        if True:
            print(f"\ntrain_generic({modality},{joint_posterior},{data_type},{device=},{folds=}) Data...",flush=True)

            print(f"Reading {data_type} ...",flush=True)
            if data_type == "alpha+beta-only":
                df = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'alpha-beta.csv'))
            elif data_type == "full":
                df1 = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'beta.csv'))
                df2 = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'alpha-beta.csv'))
                df = pd.concat([df1, df2]).reset_index()
            else:
                assert data_type == "beta-only"
                df = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'beta.csv'))

            print(f"Reading {data_type} ... DONE",flush=True)
            print(f"\ntrain_generic({modality},{joint_posterior},{data_type},{device=},{folds=}) Training...",flush=True)
            for i in range(folds):  # 'folds' independent train/test splits
                run_name = f"{run_name_base}.rep-{i}"
                ckp_name = os.path.join(RESULTS_BASE, f"{run_name}.pth")
                csv_name = os.path.join(RESULTS_BASE, f"{run_name}.csv")
                print(f"{recalculate=} {ckp_name=} {csv_name=}",flush=True)
                missing_ckp = not Path(ckp_name).is_file()
                missing_csv = not Path(csv_name).is_file()
                #print(f"os.isfile({ckp_name})={os.isfile(cpk_name)}",flush=True)
                #print(f"os.isfile({csv_name})={os.isfile(csv_name)}",flush=True)
                if recalculate or missing_ckp or missing_csv:
                    print(f"\nNeed to run {run_name}...",flush=True)
                else:
                    print(f"\nrun {run_name} SKIPPED:"
                          f"\n    {ckp_name} EXISTS"
                          f"\n    {csv_name} EXISTS\n", flush=True)
                    continue

                if dry_run:
                    print(f"\ndry run... skipping training (no output)\n",flush=True)
                    continue

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
                trainer.save_checkpoint(checkpoint, folder='./', filename=ckp_name)

                # test
                model = MVIB.from_checkpoint(checkpoint, torch.device("cpu"))
                pred = model.classify(
                    pep=ds_test.pep,
                    cdr3b=ds_test.cdr3b,
                    cdr3a=None if ds_test.cdr3a_col is None else ds_test.cdr3a)
                pred = pred.detach().numpy()
                df_test['prediction_'+str(i)] = pred.squeeze().tolist()

                # save results for further analysis
                df_test.to_csv(csv_name, index=False)
                print(f"Good: {run_name} DONE!")

            print("GOOD: mvib.{run_name_base}.* FINISHED")

def infer_generic(
    modality:str,
    joint_posterior:str,
    data_type:str,
    device:str = "cuda",
    folds:int = 5,
    verbose:int = 0,
):
    """ Do data splits 'as per training', but load checkpoint and rerun inference
        on a 'test' fold (20%).


        todo: run auc scores and such on full dataset (instead).
    """
        
    assert modality        in ['bimodal', 'trimodal', 'bimodal-alpha']
    assert joint_posterior in ['aoe', 'poe', 'max_pool', 'avg_pool']
    assert data_type       in ["alpha+beta-only", "full", "beta-only"]
    run_name_base = f"mvib.{modality}.{joint_posterior}.{data_type}" #.rep-{i}
    print(f"\ntrain_generic {run_name_base}..."
          f"\ntrain_generic({modality},{joint_posterior},{data_type},{device=},{folds=}) Data...",flush=True)
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

    if True:
        if True:
            print(f"\ntrain_generic({modality},{joint_posterior},{data_type},{device=},{folds=}) Data...",flush=True)

            print(f"Reading {data_type} ...",flush=True)
            if data_type == "alpha+beta-only":
                df = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'alpha-beta.csv'))
            elif data_type == "full":
                df1 = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'beta.csv'))
                df2 = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'alpha-beta.csv'))
                df = pd.concat([df1, df2]).reset_index()
            else:
                assert data_type == "beta-only"
                df = pd.read_csv(os.path.join(DATA_BASE, 'alpha-beta-splits', 'beta.csv'))

            print(f"Reading {data_type} ... DONE",flush=True)
            print(f"\ntrain_generic({modality},{joint_posterior},{data_type},{device=},{folds=}) Training...",flush=True)
            for i in range(folds):  # 'folds' independent train/test splits
                run_name = f"{run_name_base}.rep-{i}"
                ckp_name = os.path.join(RESULTS_BASE, f"{run_name}.pth")
                csv_name = os.path.join(RESULTS_BASE, f"infer-{run_name}.csv")
                missing_ckp = not Path(ckp_name).is_file()
                #missing_csv = not Path(csv_name).is_file()
                #print(f"os.isfile({ckp_name})={os.isfile(cpk_name)}",flush=True)
                #print(f"os.isfile({csv_name})={os.isfile(csv_name)}",flush=True)
                if missing_ckp:
                    print(f"\nSKIPPED : No checkpoint file -- need to train{run_name}...",flush=True)
                    continue
                else:
                    print(f"\nrun {run_name} INFERING:"
                          f"\n    {ckp_name} EXISTS\n"
                          #f"\n    {csv_name} EXISTS\n"
                          , flush=True)

                set_random_seed(i)

                # Change from training: put ds_test onto 'device' (GPU), since we only use this data...
                df_train, df_test = train_test_split(df.copy(), test_size=0.2, random_state=i)
                scaler = TCRDataset(df_train.copy(), device, cdr3b_col=cdr3b_col, cdr3a_col=cdr3a_col).scaler
                # here we will use ds_test, and ignore ds_train and ds_val
                ds_test = TCRDataset(df_test, device, cdr3b_col=cdr3b_col, cdr3a_col=cdr3a_col, scaler=scaler)
                if False: # skip if don't need df_train, df_val (ds_train, ds_val)  (train_loader, val_loader)
                    df_train, df_val = train_test_split(df_train, test_size=0.2, stratify=df_train.sign, random_state=i)

                    # train loader with balanced sampling
                    ds_train = TCRDataset(df_train, torch.device("cpu"), cdr3b_col=cdr3b_col,
                                          cdr3a_col=cdr3a_col, scaler=scaler)
                    class_count = np.array([df_train[df_train.sign == 0].shape[0],
                                            df_train[df_train.sign == 1].shape[0]])
                    weight = 1. / class_count
                    samples_weight = torch.tensor([weight[s] for s in df_train.sign])
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                    train_loader = torch.utils.data.DataLoader( ds_train, batch_size=batch_size, sampler=sampler)

                    # val loader with balanced sampling
                    ds_val = TCRDataset(df_val, torch.device("cpu"), cdr3b_col=cdr3b_col,
                                        cdr3a_col=cdr3a_col, scaler=scaler)
                    class_count = np.array([df_val[df_val.sign == 0].shape[0], df_val[df_val.sign == 1].shape[0]])
                    weight = 1. / class_count
                    samples_weight = torch.tensor([weight[s] for s in df_val.sign])
                    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                    val_loader = torch.utils.data.DataLoader( ds_val, batch_size=batch_size, sampler=sampler)

                model = MVIB(z_dim=z_dim, device=device, joint_posterior=joint_posterior).to(device)

                #trainer = TrainerMVIB(
                #    model, epochs=epochs, lr=lr, beta=beta, checkpoint_dir=".",
                #    mode=trainer_mode, lr_scheduler_param=lr_scheduler_param
                #)
                #checkpoint = trainer.train(train_loader, val_loader, early_stopper_patience, monitor)
                #trainer.save_checkpoint(checkpoint, folder='./', filename=ckp_name)

                # test
                print(f"loading {ckp_name} ...",end="",flush=True)
                ckp = torch.load(ckp_name, map_location='cpu')
                if verbose >= 2:
                    print(f"\n{ckp=}", flush=True)

                if verbose >= 1:
                    print(f"\n{ckp['state_dict'     ] = }", flush=True)
                    print(f"{ckp['best_val_score' ] = }", flush=True)
                    print(f"{ckp['z_dim'          ] = }", flush=True)
                    print(f"{ckp['joint_posterior'] = }", flush=True)
                    print(f"{ckp['device'         ] = }", flush=True)
                    print(f"{ckp['optimizer'      ] = }", flush=True)
                    print(f"{ckp['epoch'          ] = }", flush=True)
                    print(f"{ckp['softmax'        ] = }", flush=True)
                    print(f"{ckp['layer_norm'     ] = }", flush=True)

                print(f" model ...",end="",flush=True)
                model = MVIB.from_checkpoint(ckp, device)
                model.eval()
                model.to(device=device) # doesn't convert all to device (is there some _to_torch?) mu, logvars wrong device
                print(f" classify ...",end="",flush=True)
                with torch.inference_mode():
                    pred = model.classify(
                        pep=ds_test.pep,
                        cdr3b=ds_test.cdr3b,
                        cdr3a=None if ds_test.cdr3a_col is None else ds_test.cdr3a
                    )

                # this originally ran on cpu (36 s) but gpu (now) is faster (22 s)
                print(f" done",flush=True)
                print(f"Save: avib_infer_generic --> {csv_name}",flush=True)
                pred = pred.detach().cpu().numpy()
                pred = pred.squeeze().tolist()
                df_test['prediction_'+str(i)] = pred
                # above produces NO OUTPUT (find how vvibtcr calculates AUC, etc.)
                print(f"{pred[0:min(len(pred),5)] = }",flush=True)

                ## save results for further analysis
                df_test.to_csv(csv_name, index=False)
                print(f"Good: avib_infer_generic {run_name} DONE!",flush=True)

            print("GOOD: avib_infer_generic mvib.{run_name_base}.* FINISHED")

