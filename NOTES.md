# vibtcr
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

