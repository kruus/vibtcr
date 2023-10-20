#!/bin/bash
cd "$(dirname "$0")"

date
#
# Unfortunately, the conda environment (jupyter kernel) is HARDWIRED into the
# end of the .ipynb.
#
# This means you can edit the .json by hand, or select the correct kernel from
# the jupyter web interface.
# (or export as .py and run in the correct environment?)
#  Actually, DO use an environment, if only to ensure jupyter nbconvert can run!
echo "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda info --envs
conda deactivate
conda activate vibtcr
export | grep CONDA

f=dataset.pMHC; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb
f=avib.pMHC; echo "Running ${f}.ipynb ..."; jupyter nbconvert --execute --clear-output ${f}.ipynb

conda deactivate
date
