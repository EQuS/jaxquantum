import pytest

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp
import jaxquantum.codes as jqtb

# GKP Code
# --

def test_gkp_code():
    gkp_qubit = jqtb.GKPQubit({"N":50})

