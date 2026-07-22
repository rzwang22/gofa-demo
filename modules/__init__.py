def load_tests(_loader, tests, _pattern):
    """Keep root-level unittest discovery out of dependency-heavy runtime modules."""
    return tests
