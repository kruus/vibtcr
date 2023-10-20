#!/bin/bash
cd "$(dirname "$0")"

# Comment out what you don't need :)

#
# The README suggests DIFFERENT environments for vibtcr and tcrmodels.
# Does any notebook uses both MVIB and ERGO2 / NetTCR
#
echo "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda info --envs
conda deactivate
conda activate vibtcr
export | grep CONDA
#
# Unfortunately, the conda environment (jupyter kernel) is HARDWIRED into the
# end of the .ipynb.
#
# This means you can edit the .json by hand, or select the correct kernel from
# the jupyter web interface.
#
# (or export as .py and run in the correct environment?)
#
#  Actually, DO use an environment, if only to ensure jupyter nbconvert can run!


# dataset figure notebooks
#f=length; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb
#f=class-distribution; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb

# create avib results
# NOTE: extended to also run mvib (poe)
#f=avib; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb
f=avib.cross; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb
# folowing retrains avib (trimodel,poe) and makes predictions with missing cdr3a or cdr3b data
#f=missing-modalities; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb

# a quick figure comparing avib predictions
#f=avib.splits-comparison; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb
# Why choose '1e-6' as a regularization parameter?
#f=beta-dkl-comparison; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb

conda deactivate
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda info --envs
conda activate tcrmodels
export | grep CONDA
pip list

# These depend on having the avib results for comparison figures
# Perhaps you don't want to rerun all of these guys...
#f=baseline.beta-only; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb || exit 1
#f=baseline.alpha+beta-only; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb || exit 1
f=baseline.cross; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb || exit 1
f=baseline.full; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb || exit 1

#conda deactivate

