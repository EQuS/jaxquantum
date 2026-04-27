## Installation

**We recommend Python 3.11+.**

### Installing from PyPI

`jaxquantum` is also published on PyPI. Simply run the following code to install the package:

```bash
pip install jaxquantum
```

If you are installing on a GPU (NVIDIA, CUDA13), then run this instead:
```bash
pip install 'jaxquantum[gpu]'
```

And, on a TPU, run this:
```bash
pip install 'jaxquantum[tpu]'
```

If you face issues running JAX on your hardware, visit this page: [https://docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html)


### Installing from source

**Recommended:** As this is a rapidly evolving project, you may wish to install the latest version of `jaxquantum` from source as follows:
```bash
pip install git+https://github.com/EQuS/jaxquantum.git
```

If you are installing on a GPU (NVIDIA, CUDA13), then run this instead:
```bash
pip install 'git+https://github.com/EQuS/jaxquantum.git#egg=jaxquantum[gpu]'
```

And, on a TPU, run this:
```bash
pip install 'git+https://github.com/EQuS/jaxquantum.git#egg=jaxquantum[tpu]'
```

If you face issues running JAX on your hardware, visit this page: [https://docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html)

### Installing from source in editable mode (recommended for developers)

If you are interested in contributing to the package, please clone this repository and install this package in editable mode after changing into the root directory of this repository:
```bash
pip install -e ".[dev,docs]"
```
This will also install extras from the `dev` and `docs` flags, which can be useful when developing the package. Since this is installed in editable mode, the package will automatically be updated after pulling new changes in the repository. Again, add the `gpu` or `tpu` extra, if needed.


## Check Hardware

To check which hardware JAX is running on, run the following python code:
```python
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
print(x.device)
```
This will, for example, print out `cuda:0` if running on a GPU.

## Common Issues

### orbax-checkpoint filename or extension is too long (Windows)

***Added April 27, 2026.***

Due to a temporary fix in one of the dependencies of jaxquantum ([ref](https://github.com/google/flax/issues/5260)), you may run into this error on Windows:
```
Installing collected packages: zipp, wadler-lindig, typing-extensions, toolz, six, simplejson, PyYAML, pyparsing, pygments, psutil, protobuf, pillow, opt_einsum, numpy, nest_asyncio, msgpack, mdurl, kiwisolver, humanize, fsspec, fonttools, etils, cycler, colorama, aiofiles, absl-py, treescope, tqdm, scipy, python-dateutil, ml_dtypes, markdown-it-py, jaxtyping, contourpy, tensorstore, rich, qutip, matplotlib, jaxlib, jax, orbax-checkpoint, optax, equinox, chex, lineax, jax-tqdm, flax, optimistix, diffrax, jaxquantum
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━ 40/50 [orbax-checkpoint]ERROR: Could not install packages due to an OSError: [WinError 206] The filename or extension is too long: 'C:\\Users\\EQuS\\miniconda3\\envs\\jqt-env\\Lib\\site-packages\\orbax\\checkpoint\\experimental\\v1\\_src\\testing\\compatibility\\checkpoints\\v0_checkpoints\\composite_checkpoint\\checkpoint_metadata_missing\\pytree_checkpointable_has_metadata\\state\\array_metadatas'
```

The fix is to enable long paths in Windows, by:
1. Opening PowerShell as Administrator.
2. Run: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
3. Restart your terminal and retry `pip install jaxquantum`.

This will not be necessary once the `orbax` issue is resolved. 

### Errors installing with GPU support (Linux)

For linux users who wish to enable Nvidia GPU support, here are some steps ([ref](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu)):

1. Make sure you NVIDIA drivers by running:
   `cat /proc/driver/nvidia/version` or `sudo ubuntu-drivers list`
2. If your driver version is >= 525.60.13 then run (may be out of date with cuda13):
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

If you receive this error:
`CUDA backend failed to initialize: jaxlib/cuda/versions_helpers.cc:98: operation cuInit(0) failed: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)`

Try rebooting or running:
`sudo reboot now`

### jaxlib error after installing GPU support


```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

Following this [thread](https://github.com/jax-ml/jax/issues/29068), try running:
```
unset LD_LIBRARY_PATH
```
