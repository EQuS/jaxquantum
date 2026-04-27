
#### Old Advice (may be out of date)

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
