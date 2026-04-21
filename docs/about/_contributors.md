*This markdown document is not processed into a docs page.*

# Publishing to PyPI

Steps:
1. Update the version in `pyproject.toml` (the `version` field under `[project]`).
2. Optionally, install the package locally: `pip install --upgrade -e .`
3. Build the distribution: `python -m build`
4. Test upload to test-pypi: `twine upload --repository-url https://test.pypi.org/legacy/ dist/jaxquantum-<VERSION>.tar.gz`
5. Upload to pypi: `twine upload dist/jaxquantum-<VERSION>.tar.gz`
6. Create release on jaxquantum github: https://github.com/EQuS/jaxquantum/releases 
   1. Include package tar.
   2. Generate release notes and add additional notes manually.
