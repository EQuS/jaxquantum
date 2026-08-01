# NVIDIA GPU on Windows

JAX does not provide native Windows CUDA wheels. Use WSL2; native Windows
installations of `jaxquantum` run on CPU.

This setup was verified with Windows 11, Ubuntu 24.04 under WSL2, JAX 0.11.0,
and an RTX 4080 Super with NVIDIA driver 595.95.

## Install

In an administrator PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if Windows requests it, launch Ubuntu once, and create a Linux user.
The Windows NVIDIA driver is the only driver required. Do not install a Linux
NVIDIA driver inside WSL; confirm that the GPU is exposed with:

```bash
/usr/lib/wsl/lib/nvidia-smi
```

Create an isolated environment inside Ubuntu:

```bash
sudo apt update
sudo apt install -y python3-venv git
python3 -m venv ~/.venvs/jaxquantum-gpu
source ~/.venvs/jaxquantum-gpu/bin/activate
python -m pip install --upgrade pip

git clone https://github.com/EQuS/jaxquantum.git
cd jaxquantum
python -m pip install -e ".[gpu,tests]"
```

Keeping the checkout in the Linux filesystem gives better file and compilation
cache performance than working below `/mnt/c`.

## Verify and run

```bash
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

The output should include `gpu` and `CudaDevice(id=0)`. Then run:

```bash
python benchmarks/performance.py
python benchmarks/roofline.py
```

Monitor utilization in another WSL terminal:

```bash
watch -n 0.5 /usr/lib/wsl/lib/nvidia-smi
```

Small simulations can be faster on CPU because GPU launch overhead dominates.
GPU acceleration becomes useful when state vectors, matrix operations, or batch
dimensions are large enough to keep the device busy.

## Memory and compile cache

JAX normally preallocates most GPU memory. On a display GPU shared with Windows,
start with:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$HOME/.cache/jax"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

The first setting reduces contention with desktop applications. The second
reuses compatible compiled programs across Python processes; keep this cache in
a directory writable only by trusted users.

See the official [JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
[GPU memory guide](https://docs.jax.dev/en/latest/gpu_memory_allocation.html),
and [persistent compilation cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
for current platform details.
