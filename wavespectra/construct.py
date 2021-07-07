"""Spectra reconstruction."""
import wavespectra
from wavespectra.core.attributes import set_spec_attributes
from wavespectra.core.utils import load_function


def construct_partition(
    fit_name="jonswap", dir_name="cartwright", fit_kwargs={}, dir_kwargs={}
):
    """Fit frequency-direction E(f, d) parametric spectrum for a partition.

    Args:
        fit_name (str): Name of a valid spectral fit function, e.g. `fit_jonswap`.
        fit_kwargs (dict): Kwargs to run the `fit_name` spectral fit function.
        dir_kwargs (dict): Kwargs to run the `dir_name` directional spread function.
        dir_name (str): Name of a valid directional spreak function, e.g. `cartwright`.

    Returns:
        - efth (SpecArray): Two-dimensional, frequency-direction spectrum E(f, d) (m2/Hz/deg).

    Note:
        - Function `fit_name` must be available in main wavespectra package.
        - Function `dir_name` must be available in wavespectra.directional subpackage.

    """
    # Import spectral fit and spreading functions
    fit_func = load_function("wavespectra", fit_name, prefix="fit_")
    dir_func = load_function("wavespectra.directional", dir_name)

    # frequency spectrum
    efth1d = fit_func(**fit_kwargs)

    # Directional spreading
    spread = dir_func(**dir_kwargs)

    # Frequency-direction spectrum
    dset = efth1d * spread
    set_spec_attributes(dset)

    return dset


def reconstruct():
    pass

