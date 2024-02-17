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


# This script:
#
#   This script does a better job of recycling GPU resources.
#   Previous versions could still end up running multiple jobs
#   on the same GPU.  Here I use 'concurrent.futures', replacing
#   an earlier version using 'multiprocessing'.  The key is to
#   pass back freed execution units in the return string, so they
#   get recycled back onto the "available devices" Queue.
#
# Added to avib_train_generic.py:
#
#   dry_run:     avib trainer just logs what it would train
#   recalculate: False [default] means if checkpoint and csv outputs
#                exist, we won't train again.
#
#  Without care, jobs could swamp a GPU and things grind to a halt.
#  I saw one gpu at 100% mem and 0% GPU usage ... job somehow "stuck"
#  with no progress!


# These jobs are completely independent training tasks, so
# just use python multiprocessing...
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from avib_train_generic import train_generic

import time
import sys
from pathlib import Path
Path("results").mkdir(exist_ok=True)

doing_demo = False # True~demo, False~vibtcr training

# I have 2 devices (perhaps double if my gpu mem allows and nvtop shows < 100% GPU)
if doing_demo:
    devices = ['cpu:0','cpu:1','cpu:2','cpu:3','cpu:4','cpu:5']
else:
    devices = ['cuda:0', 'cuda:1'] #, 'cuda:0', 'cuda:1']


def demo(a,b,c,device='cpu',q=None):
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


def original_mp_Pool():
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
            if doing_demo:
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
            print(f"{type(a)=}")
            if isinstance(a,str):
                print(f"ALL JOBS SUBMITTED: job {i} {a}", flush=True)
            else: # probably AsyncResult...
                print(f"ALL JOBS SUBMITTED: job {i} {a.get()}", flush=True)

    t1 = time.time()
    for i,r in enumerate(results):
        print(i,r)

    print("total time",t1-t0,"s")

def demo2(a,b,c,qdev=None): #,qout=None):
    #global stdout
    #print(f"demo !!!") # see nothing
    # but this works...
    #sys.stdout.write(f"\ndemo {a} {b} {c} BEG !!! ... ")
    #sys.stdout.flush()
    device = qdev.get(block=True)
    sys.stdout.write(f"BEG {device} demo2({a} {b} {c}, qdev)\n")
    sys.stdout.flush()
    #sys.stdout.write(f"{device=}")
    #sys.stdout.flush()
    t = time.time()
    time.sleep(len(c)/20)
    t = time.time() - t
    #sys.stdout.write(f"\ndemo2({a}, {b}, {c}) END !!! {device} {t:.3f} s\n")
    #sys.stdout.flush()
    #qout.put(device) # tell "next" pool guy that this device is now free
    #sys.stdout.write(f" qout.put({device}) DONE\n"); sys.stdout.flush();
    return f"{device}, job sleep({len(c)/20}) : {a} {b} {c} {device} COMPLETED in {t:.2} s"

def job2(a,b,c,qdev=None):
    device = qdev.get(block=True)
    sys.stdout.write(f"BEG {device} job2({a} {b} {c}, qdev)\n")
    sys.stdout.flush()

    t = time.time()
    #train_generic(a,b,c,device=device, dry_run=True)
    train_generic(a,b,c,device=device)
    t = time.time() - t

    sys.stdout.write(f"\ntrain_generic({a}, {b}, {c}) END !!! {device} {t:.3f} s\n")
    sys.stdout.flush()
    # NB: **MUST** begin with {device}, as we parse the return string!
    return f"{device}, train_generic({a}, {b}, {c}) COMPLETED in {t} s"

def futures_pool():
    """ This way we ACTUALLY reuse the device of the last-completed job. """
    mgr = mp.Manager()
    # NOTE I think I should rewrite this with concurrent.futures.ThreadPoolExecutor,
    #      and use as_completed to get rid of finished processes.
    #      I'm wondering whether the following leaves threads active for a long
    #      time, looking at memory usage on cpu and gpu (is it constantly rising
    #      with subsequent jobs?)
    #with mp.get_context("spawn").Pool(len(devices)) as pool:
    pfx = 'demo' if doing_demo else 'vibtcr'
    with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix=pfx) as tpe:
        futures = {}
        q = mgr.Queue() # <-- this is process-safe (mp.Queue no ok for processes)
        # after this we'll put the "just finished" device back onto q
        for d in range(len(devices)):
            q.put( devices[d] )

        print(f"initial {q.qsize()=}")
        for j,r in enumerate(runs):
            device = devices[j % len(devices)]
            print(f"{j=} {doing_demo=} {r=}",flush=True)
            kwargs={'qdev':q}
            if doing_demo:
                futures[tpe.submit(demo2, *r, **kwargs)] = j    # futures supports dictionary, very handy!
            else:
                futures[tpe.submit(job2,  *r, **kwargs)] = j
        
        # Tricky: we want to reuse devices that have completed running.
        #         Queue randomized things after the initial len(devices) jobs (by quite a bit!)
        for f in as_completed(futures):
            #print(f"{futures[f]=}",flush=True)  # this is the job number (j) (but could have more info)
            j = futures[f]  # <-- job SUBMISSION index (can use to get the finished job's arguments)
            theory = devices[ j % len(devices) ]
            # But because of Queue randomization, theory can be wrong -- very bad for oversubscribing GPU
            fres = f.result()
            freed_up = fres.split(',')[0]   # <-- the ACTUAL freed up device from our future
            #
            msg = 'ok' if freed_up == theory else freed_up
            print(f"\n---END {theory} {msg:<7s} {j=} {f.result()=}", flush=True)
            q.put(freed_up)

    t1 = time.time()
    print("total time",t1-t0,"s",flush=True)

if __name__ == '__main__':
    t0 = time.time()
    if doing_demo:
        #original_mp_Pool()
        futures_pool()
    else:
        futures_pool()
        #original_mp_Pool()
# In[ ]:




