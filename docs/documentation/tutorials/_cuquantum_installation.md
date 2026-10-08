
# cuquantum backend installation (with PID) on HPC cluster

1. Run `module load miniforge` to enable conda.

2. Create and **activate** a new conda env.

3. Checkout the feature/cqt-jax branch.

4. Run `pip install -e ".[dev,gpu]"`.

5. Run `module load cuda` (needed for cuda tools and installation).

6. Run `pip install cuquantum-python-cu13`.

7. Run `pip install --no-deps cuquantum_python_jax_cu13-0.0.7.tar.gz`.

Activate the environment on an allocated GPU node before using the cuquantum backend.

## MIT ORCD setup (0.0.7 PID)

Keep the PID archive in a private directory outside the Git checkout. Replace `/private/path` below with that directory. The following installation was verified on ORCD with Python 3.11:

```bash
module load miniforge cuda cmake/3.27.9
conda create -n cqt-env python=3.11 pip
conda activate cqt-env
python -m pip install --upgrade pip 'setuptools>=77.0.3' wheel pybind11
python -m pip install 'jax[cuda13]==0.10.0' 'cuquantum-python-cu13==26.6.0'
python -m pip install -e '.[tests]'
python -m pip install --no-build-isolation --no-deps /private/path/cuquantum_python_jax_cu13-0.0.7.tar.gz
python -m pip check
```

For later sessions, load `miniforge` and `cuda`, then activate `cqt-env` on an allocated GPU node. CMake is needed only while building the PID.

For the two-GPU `example9a_sharding_init.py` test, also build the cuDensityMat MPI interface against ORCD's Open MPI (keep the output outside Git):

```bash
module load openmpi/5.0.8
MPICC=mpicc python -m pip install --no-binary=mpi4py mpi4py
tar -xzf /private/path/cuquantum_python_jax_cu13-0.0.7.tar.gz -C /private/path
pkg="$CONDA_PREFIX/lib/python3.11/site-packages/cuquantum"
mpicc -shared -std=c99 -fPIC -I"$CUDA_HOME/include" -I"$pkg/include" \
  "$pkg/distributed_interfaces/cudensitymat_distributed_interface_mpi.c" \
  -o /private/path/libcudensitymat_distributed_interface_mpi.so
export CUDENSITYMAT_COMM_LIB=/private/path/libcudensitymat_distributed_interface_mpi.so
```

With an interactive two-GPU allocation (`salloc -p mit_normal_gpu -N 1 -c 2 --mem=4G --gres=gpu:l40s:2 --time=00:10:00`), the sample passed using `mpirun --oversubscribe -n 2 --mca pml ucx python /private/path/cuquantum_python_jax_cu13-0.0.7/samples/densitymat/example9a_sharding_init.py`. Open MPI sees one slot in this interactive allocation, hence `--oversubscribe` for the two ranks.

For jaxquantum's single-process physical-GPU checks, run `pytest -q test/manual_multi_gpu/test_two_gpu_sharding.py` inside a two-GPU allocation. This directory is excluded from default pytest discovery and CI.
