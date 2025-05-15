# /Users/phionx/Github/qc/EQuS/bosonic/jax/jaxquantum/test_example.py

import pytest

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp
import qutip as qt

minimum_version_for_tests = "0.2.0"


# Version
# ========================================
def test_conversions():
    a = jqt.basis(10, 3)
    assert jqt.jqt2qt(a) == qt.basis(10, 3)
    assert jqt.jqt2qt(None) is None
# ========================================