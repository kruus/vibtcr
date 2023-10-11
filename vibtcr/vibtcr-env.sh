#!/bin/bash
#
# Usage:
#   ./vibtcr-env.sh
# Purpose:
#   Initialize a "vibtcr" conda environment
# Prerequisites: 
#   1. Install conda/mamba  (installing mamba highly recommended)
#   2. Login with conda activated AND in 'base' environment.
#            (We want to grab some of your CONDA exported variables.)
#   3. cd into the 'vibtcr/vibtcr' directory, containing this shell script
# Running notebooks:
#   4. `jupyter-lab` with py(or `conda activate vibtcr` for manual scripts)
#   5. update this script if important packages are still missing.
# Testing:
#    With mamba, a fresh RE-install with
#       conda env remove -n vibtcr
#       ./vibtcr-env.sh
#    will probably take about 1 minute.

# Does conda look OK?
if which conda; then echo "GOOD: conda is found"; else echo "OHOH. conda not in path?"; exit 1; fi
if [ -z "${CONDA_PREFIX}" ]; then
	echo "OHOH. Please activate conda and try again"
	exit 2
fi

#  Prefer MAMBA for any slow (conda install) ops
CONDA=conda
if which mamba; then CONDA=mamba; fi # prefer mamba

# allow us to "conda activate" inside a shell script
source "${CONDA_PREFIX}/etc/profile.d/conda.sh"

conda deactivate
conda activate base
# create environment "vibtcr"
#
# To clobber existing environment, run the following and then run this script again:
#	conda env remove -n vibtcr
#
if conda env list | grep -q vibtcr; then
	echo "INFO: environment vibtcr already exists"
	echo "INFO: To recreate from scratch,  conda env remove -n vibtcr"
	echo "INFO: Re-using exising vibtcr environment"
else
	${CONDA} create -n vibtcr -y -c conda-forge -c pytorch --override-channels --file=req-conda.txt
fi

conda activate vibtcr

#
# Perhaps this pinning is overkill, but just to retain EXACTLY these versions...
#
if [ ! -f ${CONDA_PREFIX}/conda-meta/pinned ]; then # first time around...
	# PIN the required package versions,
	# so future env install/update do not push these package versions forward...
	for pkg in numpy pandas scikit-learn pytorch torchvision torchaudio; do
		conda list "^${pkg}$" | tail -n+4 | awk '{print $1 " ==" $2 }' >> ${CONDA_PREFIX}/conda-meta/pinned
	done
fi
echo "vibtcr pinned versions, file ${CONDA_PREFIX}/conda-meta/pinned:"
cat "${CONDA_PREFIX}/conda-meta/pinned"
echo ""

# Now add more stuff to environment vibtcr...

pip install -e .		# install vibtcr in "edit" mode

# install jupyterlab, matplotlib (anything else you see missing trying out the notebooks)
OTHERS="jupyterlab matplotlib seaborn"
${CONDA} install -y -c conda-forge --override-channels $OTHERS

# if data/ is not there yet, unzip data.zip
if [ -f "../data.zip" -a ! -d "../data" ]; then
	(cd .. && unzip data.zip);
fi

# Finish showing a quick listing of pytorch version and pinned packages
mamba upgrade -c pytorch --override-channels pytorch

# Now exit

# on my non-cuda laptop, I can now fire up "jupyter-lab" and
# (quite slowly) run non-gpu stuff like class-distribution.ipynb
# ... well after adjusting some directories to relative paths,
#     so the notebook "just works" when running in the directory
#     structure as it is checked out from github ...
