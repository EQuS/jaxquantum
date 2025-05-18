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
    a = jqt.basis(4, 3)
    assert jqt.jqt2qt(a) == qt.basis(4, 3)
    assert jqt.jqt2qt(None) is None

    a = jqt.displace(4, jnp.array([[1.0,2,3],[4,5,6]]))
    assert jqt.jqt2qt(a)[1][2] == jqt.jqt2qt(jqt.displace(4,6))

    b = jqt.displace(4,1.2)
    assert jqt.jnp2jqt(b.data) == b
    assert jqt.jnp2jqt(b.data, dims=(4,)) == b

    c = jqt.basis(4,1)
    assert jqt.jnp2jqt(c.data, dims=(4,)) == c

    assert jqt.qt2jqt(c) == c
    assert jqt.qt2jqt(jqt.jqt2qt(c)) == c


# ========================================