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

from avib_train_generic import train_generic

import time
import sys
from pathlib import Path
Path("results").mkdir(exist_ok=True)

def demo(a,b,c,device='cpu',lock=None):
    #global stdout
    #print(f"demo !!!") # see nothing
    # but this works...
    t = time.time()
    sys.stdout.write(f"demo {a} {b} {c} {device} BEG !!!\n")
    sys.stdout.flush()
    time.sleep(len(c)/20)
    sys.stdout.write(f"demo {a} {b} {c} {device} END !!!\n")
    sys.stdout.flush()
    t = time.time() - t
    return f" job sleep({len(c)/20}) : {a} {b} {c} {device} COMPLETED in {t} s"

jobnum = 0
# oops "Lock objects should only be shared between processes through inheritance"
def job(a,b,c,q=None):
    # device must be round-robin when each job begins EXECUTING...
    device = q.get()
    t = time.time()
    sys.stdout.write(f"\ntrain_generic({a}, {b}, {c}, {device=}) STARTS")
    sys.stdout.flush()
    train_generic(a,b,c,device=device)
    sys.stdout.write(f"\ntrain_generic({a}, {b}, {c}, {device=}) FINISHED")
    sys.stdout.flush()
    t = time.time() - t
    return f"job train_generic {a} {b} {c} {device} COMPLETED in {t} s"

results=[]
def job_result(msg):
    global results
    #line = f"job FINISHED: success={async_result.successful()} {async_result.get() }\n"
    line = f"job FINISHED: {msg}\n"
    results.append(line)
    print(line, flush=True)

def job_error(huh):
    raise huh


runs = [
    ("bimodal", "aoe", "full"),
    ("trimodal", "aoe", "alpha+beta-only"),
    ("bimodal", "aoe", "alpha+beta-only"),
    ("bimodal-alpha", "aoe", "alpha+beta-only"),
    ("bimodal", "aoe", "beta-only"),
    ("bimodal", "poe", "alpha+beta-only"),
    ("trimodal", "poe", "alpha+beta-only"),
    ("bimodal-alpha", "poe", "alpha+beta-only"),
    ("bimodal", "poe", "beta-only"),
    ("bimodal", "poe", "full"),
    ("bimodal", "max_pool", "alpha+beta-only"),
    ("trimodal", "max_pool", "alpha+beta-only"),
    ("bimodal", "avg_pool", "alpha+beta-only"),
    ("trimodal", "avg_pool", "alpha+beta-only"),
]

# These jobs are completely independent training tasks, so
# just use python multiprocessing...
import multiprocessing as mp

# I have 2 devices (perhaps double if my gpu mem allows and nvtop shows < 100% GPU)
devices = ['cuda:0', 'cuda:1'] #, 'cuda:0', 'cuda:1']
if __name__ == '__main__':
    t0 = time.time()
    #mp.set_start_method('fork')
    mgr = mp.Manager()
    q = mgr.Queue() # <-- this is process-safe (mp.Queue no ok for processes)
    # NOTE I think I should rewrite this with concurrent.futures.ThreadPoolExecutor,
    #      and use as_completed to get rid of finished processes.
    #      I'm wondering whether the following leaves threads active for a long
    #      time, looking at memory usage on cpu and gpu (is it constantly rising
    #      with subsequent jobs?)
    #with mp.get_context("spawn").Pool(len(devices)) as pool:
    with mp.Pool(len(devices)) as pool:
        jobs = []
        for j,r in enumerate(runs):
            device = devices[j % len(devices)]
            if False:
                # demo ...
                jobs.append(pool.apply_async(
                    demo, args=list(r), kwds={'device':device},
                    callback=job_result, error_callback=job_error
                ))
            else:
                # training job ...
                q.put(device)
                jobs.append(pool.apply_async(
                    job, args=list(r), kwds={'q':q},
                    callback=job_result, error_callback=job_error
                ))
        
        # call get on all AsyncResults, so pool persists all jobs are done.
        for i,a in enumerate(jobs):
            print(f"ALL JOBS SUBMITTED: job {i} {a.get()}", flush=True)

    t1 = time.time()
    for i,r in enumerate(results):
        print(i,r)

    print("total time",t1-t0,"s")
# In[ ]:




