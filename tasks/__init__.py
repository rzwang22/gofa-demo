__all__ = ["GOFAFineTuneTaskWrapper", "GOFAPretrainTaskWrapper"]


def __getattr__(name):
    if name in __all__:
        from .task_wrapper import GOFAFineTuneTaskWrapper, GOFAPretrainTaskWrapper

        return {
            "GOFAFineTuneTaskWrapper": GOFAFineTuneTaskWrapper,
            "GOFAPretrainTaskWrapper": GOFAPretrainTaskWrapper,
        }[name]
    raise AttributeError(name)


def load_tests(_loader, tests, _pattern):
    """Keep root-level unittest discovery out of dependency-heavy runtime modules."""
    return tests
