def requires_metadata(func):
    """
    Decorator to ensure that the data object has metadata before executing the function.
    If metadata is missing, raises an error.
    """

    def wrapper(self, *args, **kwargs):
        if not getattr(self, "metadata", None):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no metadata. "
                f"Ensure that the data object is properly initialized with metadata before calling '{func.__name__}'."
            )
        return func(self, *args, **kwargs)

    return wrapper
