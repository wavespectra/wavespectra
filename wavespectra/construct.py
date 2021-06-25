"""Spectra reconstruction."""
import wavespectra
from wavespectra.core.attributes import set_spec_attributes
from wavespectra.core.utils import load_function


def construct_partition(fit_name, fit_kwargs, dir_kwargs, dir_name="cartwright"):
    """Fit frequency-direction E(f, d) parametric spectrum for a partition.

    Args:
        fit_name (str): Name of a valid spectral fit function, e.g. `fit_jonswap`.
        fit_kwargs (dict): Kwargs to run the `fit_name` spectral fit function.
        dir_kwargs (dict): Kwargs to run the `dir_name` directional spread function.
        dir_name (str): Name of a valid directional spreak function, e.g. `cartwright`.

    Returns:
        - efth (SpecArray): Two-dimensional, frequency-direction spectrum E(f, d) (m2/Hz/deg).

    """
    # Import spectral fit and spreading functions
    fit_func = load_function("wavespectra", fit_name, prefix="fit_")
    dir_func = load_function("wavespectra.directional_distribution", dir_name)

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


if __name__ == "__main__":

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cmocean

    farr = np.arange(0.03, 0.3, 0.001)
    darr = np.arange(0, 360, 15)
    dafreq = xr.DataArray(farr, coords={"freq": farr}, dims=("freq",), name="freq")
    dadir = xr.DataArray(darr, coords={"dir": darr}, dims=("dir",), name="dir")

    fit_name = "fit_jonswap"
    fit_kwargs = {"freq": dafreq, "hs": 2, "tp": 10}
    dir_name = "cartwright"
    dir_kwargs = {"dir": dadir, "dm": 90, "dspr": 20}

    # dset  = construct_partition(fit_name, fit_kwargs, dir_kwargs, dir_name)
    # fig = plt.figure()
    # dset.spec.plot.contourf(as_log10=False, levels=10)

    dir_kwargs = {"dir": dadir, "dm": 90, "dspr": 40}
    dset  = construct_partition(fit_name, fit_kwargs, dir_kwargs, dir_name)
    fig = plt.figure()
    dset.spec.plot.contourf(
        as_log10=False,
        levels=20,
        # cmap="pink_r",
        cmap="coolwarm",
        # cmap="terrain_r",
        # cmap="Spectral_r",
        as_period=False,
        ylim=(0, 0.15),
    )

    plt.show()
