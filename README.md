# jaxquantum
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good&color=orange"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=Shantanu&color=inactive"/>
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=EQuS&color=inactive"/>
</p>


<!-- **Documentation**: [https://github.mit.edu/pages/EQuS/python_package_template/](https://github.mit.edu/pages/EQuS/python_package_template/)  -->

## Motivation

`jaxquantum` is intended to replace QuTiP by leveraging JAX to accelerate quantum simulation tooling, such as master equatino solving. 

## Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `jaxquantum` into your conda environment.*

`jaxquantum` may soon be published on PyPI. Once it is, simply run the following code to install the package:

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
git clone git@github.mit.edu:EQuS/EQuS_template_repo.git jaxquantum
cd jaxquantum
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev]` or `pip install --upgrade '.[dev]'` (for `zsh` users).

#### Installation for Devs

If you intend to contribute to this project, please install `jaxquantum` in editable mode as follows:
```bash
git clone git@github.mit.edu:EQuS/EQuS_template_repo.git jaxquantum
cd jaxquantum
pip install -e .[dev]
```

Please use `pip install -e '.[dev]'` if you are a `zsh` user.

Installing the package in the usual non-editable mode would require a developer to upgrade their pip installation (i.e. run `pip install --upgrade .`) every time they update the package source code.

#### Building documentation locally

Set yourself up to use the `[dev]` dependencies. Then, from the command line run:
```bash
mkdocs build
```

Then, when you're ready to deploy, run:
```bash
mkdocs gh-deploy
```

## Acknowledgements

**Core Devs:** [Shantanu Jha](https://github.com/Phionx)


This project was created by the Engineering Quantum Systems group at MIT.

