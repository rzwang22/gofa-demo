__all__ = ("GOFAMistral", "GOFAMistralConfig", "TrainingArguments", "ModelArguments")


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module(".gofa", __name__), name)
