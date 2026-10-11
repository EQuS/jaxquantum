"""The normal backends must work without the private cuQuantum extension."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("missing_module", ["cuquantum", "cuquantum.densitymat.jax"])
def test_dense_and_sparse_without_cuquantum(missing_module):
    # A fresh interpreter avoids cached imports, even when this test runs in
    # a developer environment that has the private extension installed.
    code = textwrap.dedent(
        """
        import builtins
        import sys

        original_import = builtins.__import__
        missing_module = sys.argv[1]

        def without_cuquantum(name, globals=None, locals=None, fromlist=(), level=0):
            if name == missing_module or name.startswith(missing_module + "."):
                raise ModuleNotFoundError(
                    f"No module named '{missing_module}'", name=missing_module
                )
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = without_cuquantum

        import jax.numpy as jnp
        import jaxquantum as jqt

        assert jqt.sigmax().impl_type == jqt.QarrayImplType.DENSE
        assert jqt.num(3, implementation="sparse_dia").impl_type == jqt.QarrayImplType.SPARSE_DIA

        times = jnp.linspace(0.0, 0.1, 3)
        H = 0.2 * jqt.sigmax()
        psi = jqt.basis(2, 0)
        assert jqt.sesolve(H, psi, times).data.shape == (3, 2)
        assert jqt.mesolve(H, psi.to_dm(), times).data.shape == (3, 2, 2)

        try:
            jqt.sigmax(implementation="cuquantum")
        except ImportError as exc:
            assert "cuquantum backend is unavailable" in str(exc)
        else:
            raise AssertionError("An unavailable cuquantum backend was selected")
        """
    )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", code, missing_module],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
