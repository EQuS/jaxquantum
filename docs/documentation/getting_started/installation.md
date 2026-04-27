## Installation

**We recommend Python 3.11+.**

### Installing from PyPI

`jaxquantum` is published on PyPI. Run the following to install the package:

```bash
pip install jaxquantum
```

If you are installing on a GPU (NVIDIA, CUDA 13), run this instead:
```bash
pip install 'jaxquantum[gpu]'
```

On a TPU, run this:
```bash
pip install 'jaxquantum[tpu]'
```

If you face issues running JAX on your hardware, see the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html).

### Installing from source

As this is a rapidly evolving project, you may wish to install the latest version of `jaxquantum` from source as follows:
```bash
pip install git+https://github.com/EQuS/jaxquantum.git
```

If you are installing on a GPU (NVIDIA, CUDA 13), run this instead:
```bash
pip install 'git+https://github.com/EQuS/jaxquantum.git#egg=jaxquantum[gpu]'
```

On a TPU, run this:
```bash
pip install 'git+https://github.com/EQuS/jaxquantum.git#egg=jaxquantum[tpu]'
```

If you face issues running JAX on your hardware, see the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html).

### Installing from source in editable mode (recommended for developers)

If you are interested in contributing to the package, please clone this repository and install this package in editable mode after changing into the root directory of this repository:
```bash
pip install -e ".[dev,docs,tests]"
```
This installs the `dev` , `tests` and `docs` extras as well, which are useful when developing the package. Since it is installed in editable mode, the package will automatically pick up new changes pulled in the repository. Add the `gpu` or `tpu` extra if needed.

## Check Hardware

To check which hardware JAX is running on, run the following python code:
```python
import jax.numpy as jnp
x = jnp.array([1.0, 2.0, 3.0])
print(x.device)
```
This will, for example, print out `cuda:0` if running on a GPU.

## Common Issues

Often, it is useful to debug installation errors with an LLM.

### orbax-checkpoint filename or extension is too long on Windows

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

### GPU support on Linux

The `jaxquantum[gpu]` extra installs JAX with the CUDA 13 plugin, which requires NVIDIA driver version **580 or newer**. Check your driver by running `nvidia-smi` in a terminal — look for the `Driver Version:` field.

If your driver is older than 580, either upgrade it (for example, on Ubuntu via the [graphics-drivers PPA](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa)) or install the CUDA 12 build of JAX manually by following the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
