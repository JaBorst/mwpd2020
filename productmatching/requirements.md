# Requirements

## Conda

**Python3** environment using [MiniConda](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda create -n mwpd python=3
conda activate mwpd
```

## Jupyter

- https://ipywidgets.readthedocs.io/en/stable/user_install.html (used in transformers ...)

```bash
conda install -c conda-forge juypterlab
conda install -c conda-forge nodejs=12
jupyter lab clean
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1

# conda install -c conda-forge jupyter jupyter_contrib_nbextensions
# more for jupyter-notebook extensions
pip install black autopep8 flake8 isort yapf
#conda install -c conda-forge ipywidgets
```

## ML

Generic ML/plotting/math requirements

```bash
pip install tqdm
pip install pandas numpy seaborn scikit-learn scipy
pip install matplotlib
```

PyTorch:

- [PyTorch](https://pytorch.org/get-started/locally/#start-locally), [previous versions](https://pytorch.org/get-started/previous-versions/), [WHLs (!)](https://download.pytorch.org/whl/torch_stable.html)

```bash
pip install https://download.pytorch.org/whl/cu102/torch-1.5.0-cp37-cp37m-linux_x86_64.whl
pip install transformers
```

## Data

```bash
pip install jsonlines
pip install syntok
```
