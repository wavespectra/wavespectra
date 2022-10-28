"""Watershed partitioning."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.specpart import specpart
from wavespectra.core.utils import D2R, R2D, celerity
from wavespectra.core.npstats import hs_numpy, dpm_gufunc, tps_gufunc


def mom1(spectrum, dir, theta=90.0):
    """First directional moment.

    Args:
        - theta (float): angle offset.

    Returns:
        - msin (DataArray): Sin component of the 1st directional moment.
        - mcos (DataArray): Cosine component of the 1st directional moment.

    """
    dd = dir[1] - dir[0]
    cp = np.cos(np.radians(180 + theta - dir))
    sp = np.sin(np.radians(180 + theta - dir))
    msin = (dd * spectrum * sp).sum(axis=1)
    mcos = (dd * spectrum * cp).sum(axis=1)
    return msin, mcos


def combine_swells(swell_partitions, freq, dir, swells, combine_excluded=True):
    """Merge swell partitions with smallest variances.

    Args:
        - swell_partitions (list): Partitions sorted in descending order by Hs.
        - freq (1darray): Frequency array.
        - dir (1darray): Direction array.
        - swells (int): Number of swells to keep.
        - combine_excluded (bool): If True, allow combining two small partitions that
          will both be subsequently combined onto another, if False allow only
          combining onto one of the partitions that are going to be kept in the output.

    Returns:
        - combined_partitions (list): List of combined partitions.

    """
    # Peak coordinates
    tp = []
    dpm = []
    for spectrum in swell_partitions:
        spec1d = spectrum.sum(axis=1).astype("float64")
        ipeak = np.argmax(spec1d).astype("int64")
        if (ipeak == 0) or (ipeak == spec1d.size):
            tp.append(np.nan)
            dpm.append(np.nan)
        else:
            tp.append(tps_gufunc(ipeak, spec1d, freq.astype("float32")))
            dpm.append(dpm_gufunc(ipeak, *mom1(spectrum, dir)))

    # Indices of non-null partitions
    iswell = np.where(~np.isnan(tp))[0]

    # Avoid error if all partitions are null
    if iswell.size == 0:
        return swell_partitions[:swells + 1]

    # Drop null partitions
    i0 = iswell[0]
    i1 = iswell[-1] + 1
    swell_partitions = swell_partitions[i0:i1]
    tp = tp[i0:i1]
    dpm = dpm[i0:i1]

    # Recursively merge small partitions until we only have those defined by `swells`
    while i1 > swells:
        # Distances between current swell peak and all other peaks
        dx = tp[-1] - tp[:-1]
        dy = (np.array(dpm[-1]) % 360) - (np.array(dpm[:-1]) % 360)
        dy = np.minimum(dy, 360 - dy)
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # Define if merging onto closest partition from all or from the kept ones
        if not combine_excluded:
            dist = dist[:swells]

        # merge onto selected partition
        ipart = np.argmin(dist)
        # print(f"Merge partition {i1} onto partition {ipart}")
        swell_partitions[ipart] += swell_partitions[-1]

        # Drop last index
        for l in [swell_partitions, tp, dpm]:
            l.pop(-1)

        # Update counter
        i1 -= 1

    return swell_partitions


def nppart(spectrum, spectrum_smooth, freq, dir, wspd, wdir, dpt, swells=3, agefac=1.7, wscut=0.3333, merge_swells=False, combine_excluded=True):
    """Watershed partition on a numpy array.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wspd (float): Wind speed.
        - wdir (float): Wind direction.
        - dpt (float): Water depth.
        - swells (int): Number of swell partitions to compute.
        - agefac (float): Age factor.
        - wscut (float): Wind speed cutoff.
        - merge_swells (bool): Merge less energitic swell partitions onto requested
          swells according to shortest distance between spectral peaks.
        - combine_excluded (bool): If True, allow combining two small partitions that
          will both be subsequently combined onto another, if False allow only
          combining onto one of the partitions that are going to be kept in the output.

    Returns:
        - specpart (3darray): Wave spectrum partitions with shape (np, nf, nd).

    Note:
        - The smooth spectrum `spectrum_smooth` is used to run the watershed but
          partition boundaries are applied to the original spectrum.
        - The `merge_swells` option ensures spectral variance is conserved but
          could yields multiple peaks into single partitions.

    """
    part_array = specpart.partition(spectrum_smooth)

    Up = agefac * wspd * np.cos(D2R * (dir - wdir))
    windbool = np.tile(Up, (freq.size, 1)) > np.tile(
        celerity(freq, dpt)[:, np.newaxis], (1, dir.size)
    )

    ipeak = 1  # values from specpart.partition start at 1
    part_array_max = part_array.max()
    partitions_hs_swell = np.zeros(part_array_max + 1)  # zero is used for sea
    while ipeak <= part_array_max:
        part_spec = np.where(part_array == ipeak, spectrum_smooth, 0.0)

        # Assign new partition if multiple valleys and > 20 spectral bins
        __, imin = inflection(part_spec, freq, dfres=0.01, fmin=0.05)
        if len(imin) > 0:
            part_spec_new = part_spec.copy()
            part_spec_new[int(imin[0]):, :] = 0
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
            partitions_hs_swell[ipeak] = hs_numpy(part_spec, freq, dir)

        ipeak += 1

    # Concatenate together wind sea and sorted swell mappings
    sorted_swells = np.flipud(partitions_hs_swell[1:].argsort() + 1)
    last_id = part_array_max + 1 if merge_swells else swells
    parts = np.concatenate(([0], sorted_swells[:last_id]))

    # Assign spectra partitions data
    all_parts = []
    for part in parts:
        all_parts.append(np.where(part_array == part, spectrum, 0.0))

    # Merge extra swells if requested
    if merge_swells:
        swell_parts = all_parts[1:]
        swell_parts = combine_swells(swell_parts, freq, dir, swells, combine_excluded)
        all_parts = [all_parts[0]] + swell_parts

    # Extend partitions list if not enough swells
    if len(all_parts) < swells + 1:
        nullspec = np.zeros_like(spectrum)
        nmiss = (swells + 1) - len(all_parts)
        for i in range(nmiss):
            all_parts.append(nullspec)

    return np.array(all_parts)


def partition(
    dset,
    wspd="wspd",
    wdir="wdir",
    dpt="dpt",
    dset_smooth=None,
    swells=3,
    agefac=1.7,
    wscut=0.3333,
    merge_swells=False,
    combine_excluded=True,
):
    """Watershed partitioning.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.
        - wspd (xr.DataArray, str): Wind speed DataArray or variable name in dset.
        - wdir (xr.DataArray, str): Wind direction DataArray or variable name in dset.
        - dpt (xr.DataArray, str): Depth DataArray or the variable name in dset.
        - dset_smooth (xr.DataArray, xr.Dataset): Smoothed spectra for defining watershed partitions.
        - swells (int): Number of swell partitions to compute.
        - agefac (float): Age factor.
        - wscut (float): Wind speed cutoff.
        - merge_swells (bool): Merge less energitic swell partitions onto requested
          swells according to shortest distance between spectral peaks.
        - combine_excluded (bool): If True, allow combining two small partitions that
          will both be subsequently combined onto another, if False allow only
          combining onto one of the partitions that are going to be kept in the output.

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
    if dset_smooth is None:
        dset_smooth = dset
    elif isinstance(dset_smooth, xr.Dataset):
        dset_smooth = dset_smooth[attrs.SPECNAME]

    # Partitioning full spectra
    dsout = xr.apply_ufunc(
        nppart,
        dset,
        dset_smooth,
        dset.freq,
        dset.dir,
        wspd,
        wdir,
        dpt,
        swells,
        agefac,
        wscut,
        merge_swells,
        combine_excluded,
        input_core_dims=[["freq", "dir"], ["freq", "dir"], ["freq"], ["dir"], [], [], [], [], [], [], [], []],
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
