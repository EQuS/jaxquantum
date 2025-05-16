# Install contributor extra

As a contributor, you should install jaxquantum in editable mode with the `dev` and `docs` extra:

```
pip install --upgrade -e ".[dev,docs]" 
```

# Run Tests

To run tests, simply clone this repository and in its root directory run:
```
pytest
```


# Check code coverage

In the root directory of this repository, run:
```
coverage run -m pytest
```

Then, you can view a summary of the results by running:
```
coverage report
```

If you want to dive deeper, then run:
```
coverage html
```