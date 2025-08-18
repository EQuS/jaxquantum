"""
jaxquantum
"""

import os
import json

from .utils import *  # noqa
from .core import *  # noqa


with open(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "PACKAGE.json")), "r"
) as _package_file:
    package_info = json.load(_package_file)

__version__ = package_info["version"]
__author__ = package_info["authors"]
__credits__ = package_info["credits"]
