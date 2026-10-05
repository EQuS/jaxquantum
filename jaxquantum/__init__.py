"""
jaxquantum
"""

from importlib.metadata import PackageNotFoundError, version

from .core import *
from .utils import *

try:
    __version__ = version("jaxquantum")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Shantanu Jha, Shoumik Chowdhury, Gabriele Rolleri, Max Hays"
__credits__ = "EQuS"
