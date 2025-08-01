import pytest

import sys
import os

# Add the jaxquantum directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxquantum as jqt
import jax.numpy as jnp
from matplotlib import pyplot as plt

def test_plot_wigner():

    pts = jnp.linspace(-3, 3, 11)
    rho1 = jqt.basis(10, 0)
    rho2 = jqt.Qarray.from_array([jqt.basis(10, 0)])
    rho3 = jqt.Qarray.from_array([jqt.basis(10, 0), jqt.basis(10, 1)])
    rho4 = jqt.Qarray.from_array([[jqt.basis(10, 0), jqt.basis(10, 1)],
                                  [jqt.basis(10, 2), jqt.basis(10, 3)]])
    rho5 = jqt.Qarray.from_array([[[jqt.basis(10, 0), jqt.basis(10, 1)],
                                   [jqt.basis(10, 2), jqt.basis(10, 3)]],
                                  [[jqt.basis(10, 4), jqt.basis(10, 5)],
                                   [jqt.basis(10, 6), jqt.basis(10, 7)]]])
    jqt.plot_wigner(rho1, pts)
    plt.close()
    jqt.plot_wigner(rho2, pts)
    plt.close()
    jqt.plot_wigner(rho3, pts)
    plt.close()
    jqt.plot_wigner(rho4, pts)
    plt.close()
    jqt.plot_wigner(rho5, pts)
    plt.close()
    jqt.plot_wigner(rho1.to_dm(), pts)
    plt.close()
    jqt.plot_wigner(rho2.to_dm(), pts)
    plt.close()
    jqt.plot_wigner(rho3.to_dm(), pts)
    plt.close()
    jqt.plot_wigner(rho4.to_dm(), pts)
    plt.close()
    jqt.plot_wigner(rho5.to_dm(), pts)
    plt.close()

def test_plot_husimi():
    pts = jnp.linspace(-3, 3, 11)
    rho1 = jqt.basis(10, 0)
    rho2 = jqt.Qarray.from_array([jqt.basis(10, 0)])
    rho3 = jqt.Qarray.from_array([jqt.basis(10, 0), jqt.basis(10, 1)])
    rho4 = jqt.Qarray.from_array([[jqt.basis(10, 0), jqt.basis(10, 1)],
                                  [jqt.basis(10, 2), jqt.basis(10, 3)]])
    rho5 = jqt.Qarray.from_array([[[jqt.basis(10, 0), jqt.basis(10, 1)],
                                   [jqt.basis(10, 2), jqt.basis(10, 3)]],
                                  [[jqt.basis(10, 4), jqt.basis(10, 5)],
                                   [jqt.basis(10, 6), jqt.basis(10, 7)]]])
    jqt.plot_husimi(rho1, pts)
    plt.close()
    jqt.plot_husimi(rho2, pts)
    plt.close()
    jqt.plot_husimi(rho3, pts)
    plt.close()
    jqt.plot_husimi(rho4, pts)
    plt.close()
    jqt.plot_husimi(rho5, pts)
    plt.close()
    jqt.plot_husimi(rho1.to_dm(), pts)
    plt.close()
    jqt.plot_husimi(rho2.to_dm(), pts)
    plt.close()
    jqt.plot_husimi(rho3.to_dm(), pts)
    plt.close()
    jqt.plot_husimi(rho4.to_dm(), pts)
    plt.close()
    jqt.plot_husimi(rho5.to_dm(), pts)
    plt.close()


def test_plot_cf_wigner():
    pts = jnp.linspace(-3, 3, 11)
    rho1 = jqt.basis(10, 0)
    rho2 = jqt.Qarray.from_array([jqt.basis(10, 0)])
    rho3 = jqt.Qarray.from_array([jqt.basis(10, 0), jqt.basis(10, 1)])
    rho4 = jqt.Qarray.from_array([[jqt.basis(10, 0), jqt.basis(10, 1)],
                                  [jqt.basis(10, 2), jqt.basis(10, 3)]])
    rho5 = jqt.Qarray.from_array([[[jqt.basis(10, 0), jqt.basis(10, 1)],
                                   [jqt.basis(10, 2), jqt.basis(10, 3)]],
                                  [[jqt.basis(10, 4), jqt.basis(10, 5)],
                                   [jqt.basis(10, 6), jqt.basis(10, 7)]]])
    jqt.plot_cf_wigner(rho1, pts)
    plt.close()
    jqt.plot_cf_wigner(rho2, pts)
    plt.close()
    jqt.plot_cf_wigner(rho3, pts)
    plt.close()
    jqt.plot_cf_wigner(rho4, pts)
    plt.close()
    jqt.plot_cf_wigner(rho5, pts)
    plt.close()
    jqt.plot_cf_wigner(rho1.to_dm(), pts)
    plt.close()
    jqt.plot_cf_wigner(rho2.to_dm(), pts)
    plt.close()
    jqt.plot_cf_wigner(rho3.to_dm(), pts)
    plt.close()
    jqt.plot_cf_wigner(rho4.to_dm(), pts)
    plt.close()
    jqt.plot_cf_wigner(rho5.to_dm(), pts)
    plt.close()
