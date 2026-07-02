
# cuquantum backend installation (with PID) on HPC cluster

1. Run `module load miniforge` to enable conda.

2. Create and **activate** a new conda env.

3. Checkout the feature/cqt-jax branch.

4. Run `pip install -e ".[dev,gpu]"`.

5. Run `module load cuda` (needed for cuda tools and installation).

6. Run `pip install cuquantum-python-cu13`.

7. Run `pip install --no-deps cuquantum_python_jax_cu13-0.0.6.tar.gz`.

Make sure to run `module load miniforge cuda` whenever you use the cuquantum backend. Also, make sure you have actually allocated a GPU before using the cuquantum backend.  
