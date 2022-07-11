# jaxquantum
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good&color=orange"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=Shantanu&color=inactive"/>
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=EQuS&color=inactive"/>
</p>


**Documentation**: [https://github.mit.edu/pages/EQuS/python_package_template/](https://github.mit.edu/pages/EQuS/python_package_template/) 

## Motivation

This repo is a template for python package creation.

## Installation

*Conda users, please make sure to `conda install pip` before running any pip installation if you want to install `jaxquantum` into your conda environment.*

`jaxquantum` may soon be published on PyPI. Once it is, simply run the following code to install the package:

```bash
pip install jaxquantum
```
If you also want to download the dependencies needed to run optional tutorials, please use `pip install jaxquantum[dev]` or `pip install 'jaxquantum[dev]'` (for `zsh` users).


To check if the installation was successful, run:

```bash
python -c "import jaxquantum"
```

This should execute silently if installation was successful.

## Building from source

To build `jaxquantum` from source, pip install using:

```bash
git clone git@github.mit.edu:EQuS/EQuS_template_repo.git jaxquantum
cd jaxquantum
pip install --upgrade .
```

If you also want to download the dependencies needed to run optional tutorials, please use `pip install --upgrade .[dev]` or `pip install --upgrade '.[dev]'` (for `zsh` users).

#### Installation for Devs

If you intend to contribute to this project, please install `jaxquantum` in editable mode as follows:
```bash
git clone git@github.mit.edu:EQuS/EQuS_template_repo.git jaxquantum
cd jaxquantum
pip install -e .[dev]
```

Please use `pip install -e '.[dev]'` if you are a `zsh` user.

Installing the package in the usual non-editable mode would require a developer to upgrade their pip installation (i.e. run `pip install --upgrade .`) every time they update the package source code.

#### Building documentation locally

Set yourself up to use the `[dev]` dependencies. Then, from the command line run:
```bash
mkdocs build
```

Then, when you're ready to deploy, run:
```bash
mkdocs gh-deploy
```

## Acknowledgements

**Core Devs:** [Shantanu Jha](https://github.com/Phionx)


This project was created by the Engineering Quantum Systems group at MIT.



# Additional Tips

*Feel free to delete these tips!*

# Example repo
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good&color=orange"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=Morten&color=inactive"/>
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=EQuS&color=inactive"/>
</p>
This is an example of an EQuS-github repo. Whenever you start a new repo, copy the above lines into your README.md and modify accordingly -- this'll make it easy to get a sense of what's going on :rocket:

## Repo quality
We employ a rough 'stoplight' system for signifying the quality of code in a any given repo. Whenever you make a new repo, add any of the following badges to the top of your `README.md` file. Below is (rough) definitions of stoplight system currently in use. Use your own best judgement for your code :-)

<img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Caution!&color=red" style=“vertical-align:middle;”/> Here be dragons. This could be code straight from an experiment with no guarantees for its portability. Expect a collection of scripts and just barely `docstrings`
  
<img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good&color=orange"/> Decently factored code with well-documented and somewhat portable programming.
  
<img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Great!&color=brightGreen"/> High-level fully featured code with many moving parts and used broadly.


These banners generated from <a href=https://shields.io/>shields.io</a>. Raw code for example banner:
`<img src="https://img.shields.io/static/v1?style=for-the-badge&label=<LABEL TEXT>&message=<MESSAGE TEXT>&color=<COLOR>"/>`

## Progamming tips
_Always_ use docstrings and try to use type hinting (for your own sake and everyone elses!)

```python
# Example of a function with docstring
def string_concat(a: str, b: str) -> str:
    """Concatenate two strings.

    This function concatenates two strings.

    Args:
        a: first input string
        b: second input string

    Returns:
        A string that is the concatenation of the two input strings.
    """
    return a + b
```

Find an editor you're comfortable with. Here's an example of some that are used pretty broadly:
* [Atom](https://atom.io/) + [Hydrogen](Hydrogen): Hybrid between Jupyter Notebooks and barebones text editor.
* [SublimeText](https://www.sublimetext.com/): Highly extensible barebones text editor.
* [VSCode](https://code.visualstudio.com/): Popular editor somewhere in between full IDE and barebones text editor.
  * VSCode has a rich suite of features and extensions, enabling linting, debugging, version control and more!
* [PyCharm](https://www.jetbrains.com/pycharm/): Full-fledged python-specialized IDE.

## Using GitHub
There's one zillion 'git guides' out there. Here's a select few:
* [Setting up a repository](https://www.atlassian.com/git/tutorials/setting-up-a-repository)
* [Learn git branching (awesome tutorial using browser-based terminal)](https://learngitbranching.js.org)

If you want a graphical frontend for your Git browsing, there's two widely used apps:
* [GitHub Desktop](https://desktop.github.com/): The default frontend. Not super powerful, but very nice to get an understanding of git.
* [GitKraken](https://www.gitkraken.com/): More powerful (and therefore not as straightforward as GitHub Desktop). Has nice visualization tools to see branches/commits etc of a given repo.
