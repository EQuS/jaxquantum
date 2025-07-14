"""
jaxquantum
"""

import os

from .utils import * # noqa
from .core import * # noqa


with open(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "VERSION.txt")), "r"
) as _ver_file:
    __version__ = _ver_file.read().rstrip()

__author__ = "Shantanu Jha"
__credits__ = "EQuS"
