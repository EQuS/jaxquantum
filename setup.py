"""
Setup
"""
import os

from setuptools import setup, find_namespace_packages

DIST_NAME = "jaxquantum"
PACKAGE_NAME = "jaxquantum"

REQUIREMENTS = [
    "numpy",
    "matplotlib",
    "qutip",
    "jax[cpu]",
]

EXTRA_REQUIREMENTS = {
    "dev": [
        "jupyterlab>=3.1.0",
        "mypy",
        "pylint",
        "black",
        "sphinx",
        "sphinx-book-theme",
        "sphinxcontrib-napoleon",
        "Jinja2<3.1",  # https://github.com/readthedocs/readthedocs.org/issues/9038
        "myst-nb",
    ],
}

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), PACKAGE_NAME, "VERSION.txt")
)

with open(version_path, "r") as fd:
    version_str = fd.read().rstrip()

setup(
    name=DIST_NAME,
    version=version_str,
    description=DIST_NAME,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/EQuS/jaxquantum",
    author="Shantanu Jha, Shoumik Chowdhury, Max Hays",
    author_email="shantanu.rajesh.jha@gmail.com",
    license="Apache License 2.0",
    packages=find_namespace_packages(exclude=["tutorials*"]),
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
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
        "Documentation": "https://github.com/EQuS/jaxquantum",
        "Source Code": "https://github.com/EQuS/jaxquantum",
        "Tutorials": "https://github.com/EQuS/jaxquantum/tree/main/tutorials",
    },
    include_package_data=True,
)
