*This markdown document is not processed into a docs page.*

# Publishing to PyPI

Steps:
1. Update the version in `jaxquantum/PACKAGE.json`.
2. Check for meta-data errors when building the source distribution by running: `python setup.py check`.
3. Optionally, install the package locally: `pip install --upgrade -e .`
4. Create a source distribution: `python setup.py sdist` 
5. Test upload to test-pypi: `twine upload --repository-url https://test.pypi.org/legacy/ dist/jaxquantum-<VERSION>.tar.gz`
6. Upload to pypi: `twine upload dist/jaxquantum-<VERSION>.tar.gz`
7. Create release on jaxquantum github: https://github.com/EQuS/jaxquantum/releases 
   1. Include package tar.
   2. Generate release notes and add additional notes manually.
