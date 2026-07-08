"""Package-level options."""

DATASET_TRANSFORMS = "dataset_transforms"

OPTIONS = {
    DATASET_TRANSFORMS: False,
}


class set_options:
    """Set package-level options for wavespectra.

    Can be used either as a function to set options globally or as a context
    manager to set them within a controlled block, restoring the previous
    values when exiting the block.

    Currently supported options:

    - ``dataset_transforms``: If True, methods that transform the spectral
      variable such as ``interp``, ``smooth``, ``split``, ``oned`` and the
      partitioning methods return a Dataset preserving the non-spectral
      variables when called from the Dataset accessor, and the ``wspd``,
      ``wdir`` and ``dpt`` arguments of the ``hp01`` and ``track``
      partitioning methods default to the dataset variables with those names.
      If False (default), those methods return a bare spectral DataArray and
      emit a FutureWarning. This will become the default behaviour in
      wavespectra 5.0.

    Examples:
        >>> import wavespectra
        >>> wavespectra.set_options(dataset_transforms=True)  # global
        >>> with wavespectra.set_options(dataset_transforms=True):
        ...     dsout = dset.spec.oned()  # within context only

    """

    def __init__(self, **kwargs):
        self.old = {}
        for key, value in kwargs.items():
            if key not in OPTIONS:
                raise ValueError(
                    f"Argument '{key}' is not in the set of valid options "
                    f"{set(OPTIONS)}"
                )
            self.old[key] = OPTIONS[key]
        OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_value, traceback):
        OPTIONS.update(self.old)


def get_options():
    """Return a dictionary with the current package-level options."""
    return dict(OPTIONS)
