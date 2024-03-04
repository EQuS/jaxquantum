# Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `jaxquantum` into your conda environment.*

`jaxquantum` is published on PyPI. So, to install the latest version from PyPI, simply run the following code to install the package:

```bash
pip install jaxquantum
```
If you also want to download the dependencies needed to run optional tutorials, please use `pip install jaxquantum[dev,docs]` or `pip install 'jaxquantum[dev,docs]'` (for `zsh` users).

#### Building from source

To build `jaxquantum` from source, pip install using:

```bash
git clone git@github.com:EQuS/jaxquantum.git jaxquantum
cd jaxquantum
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev,docs]` or `pip install --upgrade '.[dev,docs]'` (for `zsh` users).

#### Installation for Devs

If you intend to contribute to this project, please install `jaxquantum` in editable mode as follows:
```bash
git clone git@github.com:EQuS/jaxquantum.git jaxquantum
cd jaxquantum
pip install -e .[dev, docs]
```

Please use `pip install -e '.[dev, docs]'` if you are a `zsh` user.

Installing the package in the usual non-editable mode would require a developer to upgrade their pip installation (i.e. run `pip install --upgrade .`) every time they update the package source code.

#### Install with GPU support (Linux)

For linux users who wish to enable Nvidia GPU support, here are some steps ([ref](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu)):

1. Make sure you NVIDIA drivers by running:
   `cat /proc/driver/nvidia/version` or `sudo ubuntu-drivers list`
2. If your driver version is >= 525.60.13 then run:
   `pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` otherwise, use `cuda11_pip`
3. Test that GPU support is enabled:
4. Enjoy!

***Notes:***
If you receive this error:
```
2024-02-27 14:10:45.052355: W external/xla/xla/service/gpu/nvptx_compiler.cc:742] The NVIDIA driver's CUDA version is 12.0 which is older than the ptxas CUDA version (12.3.107). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
```

Then, you should update your NVIDIA driver by running:
```
conda install cuda -c nvidia
```
