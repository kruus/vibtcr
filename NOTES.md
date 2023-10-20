# vibtcr (notes WHILE DEVELOPING vibtcr-env.sh)
### step 1: basic conda/mamba environment
I prefer to install as much as possible via conda/mamba,
leaving as little as possible managed by pip.

- `git clone https://github.com/nec-research/vibtcr.git`
- `cd vibtcr; mamba create -n vibtcr --file=requirements.txt`
- but scikit\_learn seems no longer available?
- **fix TYPO**: requirements.txt scikit\_learn --> scikit-learn (with hyphen)
- now torch 1.10.0 not found
- **fix**: torch --> pytorch  (well *temporary* fix, lol)
- **now** `mamba create -n vibtcr --file=requirements.txt` worked.
- `conda activate vibtcr`
- `cp requirements.txt req-mamba.txt` (because this worked for Step 1)
### Step 2: vibtcr
- Edit requirements.txt and **undo** an edit: `pytorch` --> `torch`
- `conda activate vibtcr`
- **now** `pip install -e .` worked.
    - ("-e" so I can edit the vibtcr source files)
```
# simple test.  Does this run without bombing?
import vibtcr, vibtcr.mvib, vibtcr.base
from vibctr.mvib.mvib import MVIB
foo=MVIB(z_dim=10, device="cpu")

```
### Step 3: flesh out environment "vibtcr"
- in top directory, `unzip data.zip` seemed a good thing to do.
- for .ipynb, `mamba install jupyterlab`

(stopping, because cuda device is hard-wired into things, and I'm on my laptop)

### other...
- `mamba install conda-forge::nb_conda_kernels`
- ipywidgdets (tqdm wants it when running under jupyter-lab)

# tcrmodels
Needed for all the 'baselinei\*.ipynb' and perhaps some others.

These use tensorflow, so create a separate environment...

Based on the tcrmodels/requirements.txt (and adding jupyterlab support) I
found tensorflow==1.15.0 is tricky to satisfy, so I tried
getting "whatever" tensorflow and pytorch versions...

`mamba create -n tcrmodels python=3.7 keras==2.3.0 tensorflow matplotlib pandas 'pytorch-lightning==0.8.5' 'scikit-learn==0.22.1' seaborn jupyterlab ipywidgets conda-forge::nb_conda_kernels --override-channels -c conda-forge -c pytorch`

This gave python=3.7 pytorch=1.4 tensorflow==2.4.3 in a "first version" environment.  
I add --override-channels to avoid my somewhat strange .condarc channel list.

- `conda activate tcrmodels`
- `pip install -e .`

Above now **replaces** tensorflow 2.4.3 with tensorflow==1.15.0 and installs gast==0.2.2 tensorboard==1.15.0 tensorflow-estimator==1.15.1
This seems to satisfy the requirements, as well as have many things installed by conda rather than pip.

# Run jupyter-lab remotely
1. In server "screen" session, `jupyter-lab --no-browser`
2. On laptop terminal, `ssh -N -L 8888:localhost:888 snake10
3. On laptop browser, open the "127.0.0.1:8888" jupyter url

## runall.sh
### Run .ipynb from command line, without jupyter-lab server or browser
Run inside screen or tmux for long, unattended training.  
`runall.sh` clears notebook outputs and reruns all the uncommented notebooks.

### runall.sh issues
After installing `nb_conda_kernels` **jupyter-lab** should see env::vibtcr and env::tcrmodels.  
*But* sometimes `jupyter nbconvert` does not see the 'env::FOO' conda environments.
i.e. `jupyter kernelspec list` does not show the conda kernels that jupyter-lab sees!

This can be fixed manually, per kernel:  
`conda activate tcrmodels`  # it has jupyterlab ipython ipywidgets etc. already installed  
`ipython kernel install --user --name=conda-env-tcrmodels-py`  
`conda deactivate`

Then `jupyter kernelspec list` should have an environment matching the name near the
end of JSON-format `baseline-FOO.ipynb`.

Restart jupyter-lab  
edit ([un]comment) and try `runall.sh` again.

**Now multiple attempts fixing errors** creating log files runall[1..5].sh
were run until notebook pre-dependencies (in 'results/') were "complete".

- Some additional output in *results/* were required, modifying avib.py to run
them (and then rerun various dependent *baseline*-FOO notebooks).
- **typo:** avib.cross.ipynb (not avib-cross) typo: needed rerun to create
results/ for `baseline.cross.ipynb`


