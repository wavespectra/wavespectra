"""Watershed partitioning."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.specpart import specpart
from wavespectra.core.utils import D2R, R2D, celerity
from wavespectra.core.npstats import hs, dpm_gufunc, tps_gufunc
from wavespectra.partition.utils import combine_partitions


def np_ptm1(
    spectrum,
    spectrum_smooth,
    freq,
    dir,
    wspd,
    wdir,
    dpt,
    agefac=1.7,
    wscut=0.3333,
    swells=None,
    combine=False
):
    """PTM1 spectra partitioning.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wspd (float): Wind speed.
        - wdir (float): Wind direction.
        - dpt (float): Water depth.
        - agefac (float): Age factor.
        - wscut (float): Wind sea fraction cutoff.
        - swells (int): Number of swell partitions to compute, all detected by default.
        - combine (bool): Combine less energitic partitions onto one of the keeping
          ones according to shortest distance between spectral peaks.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd).

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.
        - The `combine` option ensures spectral variance is conserved but
          could yields multiple peaks into single partitions.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth)
    nparts = watershed_map.max()

    # Wind sea mask
    up = np.tile(agefac * wspd * np.cos(D2R * (dir - wdir)), (freq.size, 1))
    windseamask = up > np.tile(celerity(freq, dpt)[:, np.newaxis], (1, dir.size))

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_partition = np.zeros_like(spectrum)
    swell_partitions = [np.zeros_like(spectrum) for n in range(nparts)]
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0) # start at 1
        wsfrac = part[windseamask].sum() / part.sum()
        if wsfrac > wscut:
            wsea_partition += part
        else:
            swell_partitions[ipart] += part

    # Sort swells by Hs
    isort = np.argsort([-hs(swell, freq, dir) for swell in swell_partitions])
    swell_partitions = [swell for _, swell in sorted(zip(isort, swell_partitions))]

    # Dealing with the number of swells
    if swells is None:
        # Exclude null swell partitions if the number of output swells is undefined
        swell_partitions = [swell for swell in swell_partitions if swell.sum() > 0]
    else:
        if nparts > swells and combine:
            # Combine extra swell partitions into main ones
            swell_partitions = combine_partitions(swell_partitions, freq, dir, swells)
        elif nparts > swells and not combine:
            # Discard extra partitions
            swell_partitions = swell_partitions[:swells]
        elif nparts < swells:
            # Extend partitions list with null spectra
            n = swells - len(swell_partitions)
            for i in range(n):
                swell_partitions.append(np.zeros_like(spectrum))

    return np.array([wsea_partition] + swell_partitions)


def np_ptm3(spectrum, spectrum_smooth, freq, dir, parts=None, combine=False):
    """PTM3 spectra partitioning.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - parts (int): Number of partitions to compute, all detected by default.
        - combine (bool): Combine less energitic partitions onto one of the keeping
          ones according to shortest distance between spectral peaks.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd).

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.
        - The `combine` option ensures spectral variance is conserved but
          could yields multiple peaks into single partitions.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth)
    nparts = watershed_map.max()

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    partitions = []
    for npart in range(1, nparts + 1):
        partitions.append(np.where(watershed_map == npart, spectrum, 0.0))

    # Sort partitions by Hs
    hs_partitions = [hs(partition, freq, dir) for partition in partitions]
    partitions = [p for _, p in sorted(zip(hs_partitions, partitions), reverse=True)]

    if parts is not None:
        if nparts > parts and combine:
            # Combine extra partitions into main ones
            partitions = combine_partitions(partitions, freq, dir, parts)
        elif nparts > parts and not combine:
            # Discard extra partitions
            partitions = partitions[:parts]
        elif nparts < parts:
            # Extend partitions list with zero arrays
            template = np.zeros_like(spectrum)
            n = parts - len(partitions)
            for i in range(n):
                partitions.append(template)

    return np.array(partitions)


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
            partitions_hs_swell[ipeak] = hs(part_spec, freq, dir)

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
