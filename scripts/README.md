Here I transform some of the original notebooks into script form.
This reduces a lot of copy-paste that went on in the training notebooks.

### todo
[x] Add some scripts to quickly test model loading and inference, particularly
when testing upstream pytorch/python packages needed to play nicely in a
"modern" tcrppo-internal environment.
[x] Run trained models in 'tcr6' tcrppo-internal environment.
    - see perhaps 10% speedup under pytorch 2.0
    - rerunning 'test' gets similar outputs, +/- 2--3 %
      (not exact because of avib stochastic sampling)
[ ] Try out 'ray' to run multiple jobs on all available (local) gpus.

