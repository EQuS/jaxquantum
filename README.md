# jaxquantum
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good&color=orange"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=Shantanu&color=inactive"/>
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=EQuS&color=inactive"/>
</p>

## Motivation

`jaxquantum` leverages JAX to enable the auto differentiable and accelerated simulation of quantum dynamical systems, through tooling such as master equation solving. 

## Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `jaxquantum` into your conda environment.*

`jaxquantum` is published on PyPI. So, to install the latest version from PyPI, simply run the following code to install the package:

```bash
pip install jaxquantum
```
If you also want to download the dependencies needed to run optional tutorials, please use `pip install jaxquantum[dev]` or `pip install 'jaxquantum[dev]'` (for `zsh` users).


To check if the installation was successful, run:

```bash
python -c "import jaxquantum"
```

This should execute silently if installation was successful.

## Building from source

To build `jaxquantum` from source, pip install using:

```bash
git clone git@github.com:EQuS/jaxquantum.git jaxquantum
cd jaxquantum
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev]` or `pip install --upgrade '.[dev]'` (for `zsh` users).

#### Installation for Devs

If you intend to contribute to this project, please install `jaxquantum` in editable mode as follows:
```bash
git clone git@github.com:EQuS/jaxquantum.git jaxquantum
cd jaxquantum
pip install -e .[dev, docs]
```

Please use `pip install -e '.[dev, docs]'` if you are a `zsh` user.

Installing the package in the usual non-editable mode would require a developer to upgrade their pip installation (i.e. run `pip install --upgrade .`) every time they update the package source code.

## Documentation

Documentation should be viewable here: [https://github.com/pages/EQuS/jaxquantum/](https://github.com/pages/EQuS/jaxquantum/) 

### Build and view locally

To view documentation locally, plesae make sure the install the requirements under the `docs` extra, as specified above. Then, run the following:

```
mkdocs serve
```

The documentation should now be at the url provided by the above command. 

### Updating Docs

The documentation should be updated automatically when any changes are made to the `main` branch. However, updates can also be forced by running:

```
mkdocs gh-deploy --force
```
This will build your documentation and deploy it to a branch gh-pages in your repository.

## Acknowledgements

**Core Devs:** [Shantanu Jha](https://github.com/Phionx), [Shoumik Chowdhury](https://github.com/shoumikdc)


This package was developed while conducting research in the Engineering Quantum Systems Group at MIT with invaluable advice from [Prof. William D. Oliver](https://equs.mit.edu/william-d-oliver/). 

