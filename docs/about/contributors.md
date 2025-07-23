# Contributions & Contact

This package is open source and, as such, very open to contributions. Please don't hesitate to open an issue, report a bug, request a feature, or create a pull request. We are also open to deeper collaborations to create a tool that is more useful for everyone. If a discussion would be helpful, please email [shanjha@mit.edu](mailto:shanjha@mit.edu) to set up a meeting. 

---

# Tests

### Install contributor extra

As a contributor, you should install jaxquantum in editable mode with the `dev` and `docs` extra:

```
pip install --upgrade -e ".[dev,docs]" 
```

### Run Tests

To run tests, simply clone this repository and in its root directory run:
```
pytest
```

### Check code coverage

In the root directory of this repository, run:
```
coverage run --source=jaxquantum -m pytest
```

Then, you can view a summary of the results by running:
```
coverage report
```

If you want to dive deeper, then run:
```
coverage html
```