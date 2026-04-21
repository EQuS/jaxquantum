"""
jaxquantum
"""

from importlib.metadata import version, PackageNotFoundError

from .utils import *  # noqa
from .core import *  # noqa

try:
    __version__ = version("jaxquantum")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Shantanu Jha, Shoumik Chowdhury, Gabriele Rolleri, Max Hays"
__credits__ = "EQuS"
