"""
Setup
"""
import os

from setuptools import setup, find_namespace_packages
import json

DIST_NAME = "jaxquantum"
PACKAGE_NAME = "jaxquantum"

REQUIREMENTS = [
    "numpy",
    "matplotlib",
    "qutip",
    "jax[cpu]",
    "diffrax",
    "flax",
    "tqdm",
    "optax",
    "jax-tqdm"
]

EXTRA_REQUIREMENTS = {
    "dev": [
        "jupyterlab>=3.1.0",
        "mypy",
        "pylint",
        "black",
        "coverage"
    ],
    "docs": [
        "mkdocs",
        "mkdocs-material",
        "mkdocs-literate-nav",
        "mkdocs-section-index",
        "mkdocs-gen-files",
        "mkdocstrings-python",
        "mkdocs-jupyter",
        "pymdown-extensions"
    ],
    "tests": [
        "pytest",
        "pytest-cov"
    ],
    "gpu": [
        "jax[cuda13]" # only officially supported on linux
    ],
    "tpu": [
        "jax[tpu]" # only officially supported on linux
    ]
}

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

package_info_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "PACKAGE.json")
)

with open(package_info_path, "r") as fd:
    package_info = json.load(fd)

setup(
    name=DIST_NAME,
    version=package_info["version"],
    description=DIST_NAME,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://jaxquantum.org",
    author=package_info["authors"],
    author_email="shantanu.rajesh.jha@gmail.com",
    license="Apache-2.0",
    packages=find_namespace_packages(exclude=["experiments*"]),
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    keywords="quantum computing",
    python_requires=">=3.7",
    project_urls={
        "Documentation": "https://jaxquantum.org",
        "Source Code": "https://github.com/EQuS/jaxquantum",
        "Tutorials": "https://github.com/EQuS/jaxquantum/tree/main/tutorials",
    },
    include_package_data=True,
)
