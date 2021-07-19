"""Spectra reconstruction."""
import numpy as np
import xarray as xr
import wavespectra
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import load_function


# Wave stats to use with reconstruction
STATS = [
    "hs",
    "tp",
    "fp",
    "dm",
    "dpm",
    "dspr",
    "dpspr",
    "tm01",
    "tm02",
    "gamma",
    "alpha",
]


def construct_partition(
    fit_name="jonswap", dir_name="cartwright", fit_kwargs={}, dir_kwargs={}
):
    """Fit frequency-direction E(f, d) parametric spectrum for a partition.

    Args:
        - fit_name (str): Name of a valid spectral fit function, e.g. `fit_jonswap`.
        - dir_name (str): Name of a valid directional spreak function, e.g. `cartwright`.
        - fit_kwargs (dict): Kwargs to run the `fit_name` spectral fit function.
        - dir_kwargs (dict): Kwargs to run the `dir_name` directional spread function.

    Returns:
        - efth (SpecArray): Two-dimensional, frequency-direction spectrum E(f, d) (m2/Hz/deg).

    Note:
        - Function `fit_name` must be available in main wavespectra package.
        - Function `dir_name` must be available in wavespectra.directional subpackage.
        - Missing values in output spectrum are filled with zeros.

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

    return dset.fillna(0.)


def partition_and_reconstruct(
    dset,
    swells=4,
    fit_name="fit_jonswap",
    dir_name="cartwright",
    method_combine="max",
):
    """Partition and reconstruct existing spectra to evaluate.

    Args:
        - dset (SpecDataset): Spectra object to partition and reconstruct.
        - swells (int): Number of swell partitions to use in reconstruction.
        - fit_name (str, list): Name of a valid fit function, e.g. `fit_jonswap`, or a
          list of names with len=`swells`+1 to define one fit function for the wind sea
          and each swell partition.
        - dir_name (str, list): Name of a valid directional spread function, e.g.
          `cartwright`, or a list of names with len=`swells`+1 to define one
          directional spread for the wind sea and each swell partition.
        - method_combine (str): Method to combine partitions.

    Returns:
        - dsout (SpecArray): Reconstructed spectra with same coordinates as dset.

    Note:
        - If `fit_name` or `dir_name` are str, the functions specified by these
          arguments are applied to all sea and swell partitions.

    """
    # Parameter checking
    if isinstance(fit_name, str):
        fit_name = (swells + 1) * [fit_name]
    if isinstance(dir_name, str):
        dir_name = (swells + 1) * [dir_name]
    for name in [fit_name, dir_name]:
        if len(name) != swells + 1:
            raise ValueError(
                f"Len of '{name}' must correspond to the "
                f"number of wave systems '{swells + 1}'"
            )

    coords = {attrs.FREQNAME: dset[attrs.FREQNAME], attrs.DIRNAME: dset[attrs.DIRNAME]}

    # Partitioning
    dspart = dset.spec.partition(dset.wspd, dset.wdir, dset.dpt, swells=swells)

    # Calculating parameters
    dparam = dspart.spec.stats(STATS)

    # Reconstruct partitions
    reconstructed = []
    for ipart, part in enumerate(dspart.part):

        # Turn partitioned parameters for current partition into functions kwargs
        kw = {**coords, **{k: v for k, v in dparam.sel(part=[part]).data_vars.items()}}

        # Reconstruct current partition
        reconstructed = construct_partition(
            fit_name=fit_name[ipart],
            dir_name=dir_name[ipart],
            fit_kwargs=kw,
            dir_kwargs=kw
        )

    # Combine partitions
    reconstructed = getattr(xr.concat(reconstructed, attrs.PARTNAME), method_combine)(attrs.PARTNAME)
    reconstructed = reconstructed.to_dataset(name=attrs.SPECNAME)
    set_spec_attributes(reconstructed)

    # Add back winds and depth
    reconstructed["wspd"] = dset.wspd
    reconstructed["wdir"] = dset.wdir
    reconstructed["dpt"] = dset.dpt

    # Set some attributes
    reconstructed.attrs = {
        "title": "Spectra Reconstruction",
        "source": "wavespectra <https://github.com/wavespectra/wavespectra>",
        "partitions": swells + 1,
        "spectral_shapes": fit_name,
        "directional_spread": dir_name,
    }
    return reconstructed

