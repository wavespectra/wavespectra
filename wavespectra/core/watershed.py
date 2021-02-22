"""Watershed partitioning."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.specpart import specpart
from wavespectra.core.utils import D2R, R2D, celerity
from wavespectra.core.npstats import hs


def nppart(spectrum, freq, dir, wspd, wdir, dpt, swells=3, agefac=1.7, wscut=0.3333):
    """Watershed partition on a numpy array.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wspd (float): Wind speed.
        - wdir (float): Wind direction.
        - dpt (float): Water depth.
        - swells (int): Number of swell partitions to compute.
        - agefac (float): Age factor.
        - wscut (float): Wind speed cutoff.

    Returns:
        - specpart (3darray): Wave spectrum partitions with shape (np, nf, nd).

    """
    part_array = specpart.partition(spectrum)

    Up = agefac * wspd * np.cos(D2R * (dir - wdir))
    windbool = np.tile(Up, (freq.size, 1)) > np.tile(
        celerity(freq, dpt)[:, np.newaxis], (1, dir.size)
    )

    ipeak = 1  # values from specpart.partition start at 1
    part_array_max = part_array.max()
    # partitions_hs_swell = np.zeros(part_array_max + 1)  # zero is used for sea
    partitions_hs_swell = np.zeros(part_array_max + 1)  # zero is used for sea
    while ipeak <= part_array_max:
        part_spec = np.where(part_array == ipeak, spectrum, 0.0)

        # Assign new partition if multiple valleys and satisfying conditions
        __, imin = inflection(part_spec, freq, dfres=0.01, fmin=0.05)
        if len(imin) > 0:
            part_spec_new = part_spec.copy()
            part_spec_new[imin[0].squeeze() :, :] = 0
            newpart = part_spec_new > 0
            if newpart.sum() > 20:
                part_spec[newpart] = 0
                part_array_max += 1
                part_array[newpart] = part_array_max
                partitions_hs_swell = np.append(partitions_hs_swell, 0)

        # Assign sea partition
        W = part_spec[windbool].sum() / part_spec.sum()
        if W > wscut:
            part_array[part_array == ipeak] = 0
        else:
            partitions_hs_swell[ipeak] = hs(part_spec, freq, dir)

        ipeak += 1

    sorted_swells = np.flipud(partitions_hs_swell[1:].argsort() + 1)
    parts = np.concatenate(([0], sorted_swells[:swells]))
    all_parts = []
    for part in parts:
        all_parts.append(np.where(part_array == part, spectrum, 0.0))

    # Extend partitions list if it is less than swells
    if len(all_parts) < swells + 1:
        nullspec = 0 * spectrum
        nmiss = (swells + 1) - len(all_parts)
        for i in range(nmiss):
            all_parts.append(nullspec)

    return np.array(all_parts)


def partition(
    dset,
    wspd="wspd",
    wdir="wdir",
    dpt="dpt",
    swells=3,
    agefac=1.7,
    wscut=0.3333,
):
    """Watershed partitioning.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.
        - wspd (xr.DataArray, str): Wind speed DataArray or variable name in dset.
        - wdir (xr.DataArray, str): Wind direction DataArray or variable name in dset.
        - dpt (xr.DataArray, str): Depth DataArray or the variable name in dset.
        - swells (int): Number of swell partitions to compute.
        - agefac (float): Age factor.
        - wscut (float): Wind speed cutoff.

    Returns:
        - dspart (xr.Dataset): Partitioned spectra dataset with extra dimension.

    References:
        - Hanson, Jeffrey L., et al. "Pacific hindcast performance of three
            numerical wave models." JTECH 26.8 (2009): 1614-1633.

    """
    # Sort out inputs
    if isinstance(wspd, str):
        wspd = dset[wspd]
    if isinstance(wdir, str):
        wdir = dset[wdir]
    if isinstance(dpt, str):
        dpt = dset[dpt]
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]

    # Partitioning full spectra
    dsout = xr.apply_ufunc(
        nppart,
        dset,
        dset.freq,
        dset.dir,
        wspd,
        wdir,
        dpt,
        swells,
        agefac,
        wscut,
        input_core_dims=[["freq", "dir"], ["freq"], ["dir"], [], [], [], [], [], []],
        output_core_dims=[["part", "freq", "dir"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={
            "output_sizes": {
                "part": swells + 1,
            },
        },
    )

    # Finalise output
    dsout.name = "efth"
    dsout["part"] = np.arange(swells + 1)
    dsout.part.attrs = {"standard_name": "spectral_partition_number", "units": ""}

    return dsout.transpose("part", ...)


def frequency_resolution(freq):
    """Frequency resolution array.

    Args:
        - freq (1darray): Frequencies to calculate resolutions from with shape (nf).

    Returns:
        - df (1darray): Resolution for pairs of adjacent frequencies with shape (nf-1).

    """
    if len(freq) > 1:
        return abs(freq[1:] - freq[:-1])
    else:
        return np.array((1.0,))


def inflection(spectrum, freq, dfres=0.01, fmin=0.05):
    """Points of inflection in smoothed frequency spectra.

    Args:
        - fdspec (ndarray): freq-dir 2D specarray.
        - dfres (float): used to determine length of smoothing window.
        - fmin (float): minimum frequency for looking for minima/maxima.

    """
    if len(freq) > 1:
        df = frequency_resolution(freq)
        sf = spectrum.sum(axis=1)
        nsmooth = int(dfres / df[0])  # Window size
        if nsmooth > 1:
            sf = np.convolve(sf, np.hamming(nsmooth), "same")  # Smoothed f-spectrum
        sf[(sf < 0) | (freq < fmin)] = 0
        diff = np.diff(sf)
        imax = np.argwhere(np.diff(np.sign(diff)) == -2) + 1
        imin = np.argwhere(np.diff(np.sign(diff)) == 2) + 1
    else:
        imax = 0
        imin = 0
    return imax, imin
