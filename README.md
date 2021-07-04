# scIHPF

single-cell Integrative Hierarchical Poisson Factorisation

Python package for data integration in scRNA-seq datasets extending Hierarchical Poisson Factorisation model by Blei et al. (2017). 



## Related packages 

- scHPF: https://github.com/simslab/scHPF
- scanorama: https://github.com/brianhie/scanorama


### Installation instructions 

1. Create a conda environment as follows 

conda create -n ihpf
conda activate ihpf

conda install python=3.7
conda install -c anaconda ipykernel
conda install seaborn scikit-learn statsmodels numba pytables
python -m pip install scanpy --no-deps
python -m pip install schpf --no-deps
python -m ipykernel install --user --name=ihpf

2. Clone the repo and install 

git clone https://github.com/barahona-research-group/scIHPF.git
cd scIHPF
python -m pip install . 