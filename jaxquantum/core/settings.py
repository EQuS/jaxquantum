"""Core settings.

``default_backend`` is stored as a string (matching ``QarrayImplType`` values)
to avoid a circular import with ``qarray.py``.  ``Qarray.create`` resolves it
to a ``QarrayImplType`` member at call time.
"""

SETTINGS = {
    "auto_tidyup_atol": 1e-14,
    "default_backend": "dense",
}
