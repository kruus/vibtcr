# vibtcr
`vibtcr` is a Python package which implements the Mutlimodal Variational Information 
Bottleneck (MVIB) and the Attentive Variational Information Bottleneck (AVIB), with a focus on
TCR-peptide interaction prediction.

Paper: ["Attentive Variational Information Bottleneck for TCR–peptide interaction prediction"](https://doi.org/10.1093/bioinformatics/btac820),  F. Grazioli, P. Machart, A. Mösch, K. Li, L.V. Castorina, N. Pfeifer, M.R. Min, Bioinformatics 2022

![architecture](architecture.png?raw=true "AVIB architecture")

## Install `vibtcr`
```
cd vibtcr
pip install .
```
Remark: `vibtcr` requires a different version of PyTorch than `tcrmodels`. It's recommended to install them in different environments.

#### OLD conda install for pytorch 1.10 (see NOTES.md for more details)
- after a light edit of requirements.txt (NOTES.md), `mamba create -n vibtcr --file=requirements-orig.txt`
- [opt] to run original notebooks:
	- `conda activate vibtcr; mamba install jupyterlab conda-forge::nb_conda_kernels`ipywidgets`
- but there is still some confusion about tensorflow requirements, and another install path is described too.
- *original `vibtcr` env might be required if you intend to run the ERGO II and NetTCR2.0 external codes*

#### UPSTREAM conda packages for tcrppo integration (back story, WIP)
- However, dumbing down requirements.txt to be unversioned, I can activate a modern pytorch 2.0 env
- as described in github `tcrppo-internal` (NECLA gitlab project under Kai Li's name, I think)
- Now a `pip install .` for vibtcr installs NO additional packages (as env 'tcr6' already has them)
- and a simple inference script `scripts/classification:$ python avib-infer0.py 2>&1 | tee avi0.log`
- now runs with identical log output in env `tcr6` as the original env `vibtcr`.
- the above test also runs faster (17 s now; 21 s before)

- Note: this suggests that tensorflow might not be needed to run avib itself, as vibtcr simplifies
  the conda requirements to just tensorboard (tensorboardx?), IIRC.

- Anyhow, I moved the original `requirements.txt` --> `requirements-orig.txt` and for reference have
  duplicated the `tcr6` install file here (copied Feb 2024 from tcrppo-internal)

### UPSTREAM packages for tcrppo integration (pytorch 2.x)
- follow copied instructions in `env-tcr6.md` using package file `env-tcr6.yml`
- Add to this a `pip install -e .`
    - at least until I integrate vibtcr into tcrppo-internal
- Now `conda activate tcr6` and try out some scripts
- **Caveat**: we might not be able to run the vibtcr external codes for ERGO II and NetTCR2.0
              in the `tcr6` environment.
- So only run *vanilla* vibtcr training/inference in env `tcr6`, please.

# Content
```
vibtcr
│   README.md
│   ... 
│   
└───data/
│   │   alpha-beta-splits/ (all TCR data, split in two disjoint sets: alpha+beta, beta-only)
│   │   ergo2-paper/ (data used in the ERGO II paper - contains VDJDB and McPAS)
│   │   mhc/ (the NetMHCIIpan-4.0 data)
│   │   nettcr2-paper/ (data used in the NetTCR2.0 paper - contains IEDB, VDJDB and MIRA)
│   │   vdjdb/ (complete VDJdb data from 5th of September)
│   
└───notebooks/
│   │   notebooks.classification/ (TCR-peptide experiments with AVIB/MVIB)
|   │   notebooks.mouse/ (identifying mouse TCRs as suitable OOD dataset)
│   │   notebooks.ood/ (out-of-distribution detection experiments with AVIB)
│   │   notebooks.regression/ (peptide-MHC BA regression with AVIB)
│   
└───tcrmodels/ (Python package which wraps SOTA ML-based TCR models)
│   
└───vibtcr/ (Python package which implements MVIB and AVIB for TCR-peptide interaction prediction)
│   
└───scripts/           plain python ( NO jupyter notebook requirement )
    |  classification/ train everything and test model load classification, WIP

```

# tcrmodels
`tcrmodels` wraps state-of-the-art ML-based TCR prediction models.
So far, it includes:
* [ERGO II](https://github.com/IdoSpringer/ERGO-II)
* [NetTCR2.0](https://github.com/mnielLab/NetTCR-2.0)

### Install `tcrmodels`
```
cd tcrmodels
pip install .
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

`tcrmodels` requires Python 3.6

### ERGO II
Springer I, Tickotsky N and Louzoun Y (2021), Contribution of T Cell Receptor Alpha and Beta CDR3, MHC Typing, V and J Genes to Peptide Binding Prediction. Front. Immunol. 12:664514. DOI: https://doi.org/10.3389/fimmu.2021.664514

### NetTCR-2.0
Montemurro, A., Schuster, V., Povlsen, H.R. et al. NetTCR-2.0 enables accurate prediction of TCR-peptide binding by using paired TCRα and β sequence data. Commun Biol 4, 1060 (2021). DOI: https://doi.org/10.1038/s42003-021-02610-3

# License
For `vibtcr`, we provide a non-commercial license, see LICENSE.txt

# Cite
If you find this work useful, please cite:
```
@article{10.1093/bioinformatics/btac820,
    author = {Grazioli, Filippo and Machart, Pierre and Mösch, Anja and Li, Kai and Castorina, Leonardo V and Pfeifer, Nico and Min, Martin Renqiang},
    title = "{Attentive Variational Information Bottleneck for TCR–peptide interaction prediction}",
    journal = {Bioinformatics},
    volume = {39},
    number = {1},
    year = {2022},
    month = {12},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btac820},
    url = {https://doi.org/10.1093/bioinformatics/btac820},
    note = {btac820},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/39/1/btac820/48493569/btac820.pdf},
}
```
