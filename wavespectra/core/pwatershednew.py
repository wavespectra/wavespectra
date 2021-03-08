"""Watershed partitioning."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core.utils import D2R, R2D, celerity
from wavespectra.core.npstats import hs


def ptnghb(nk, nth):
    """Short description.

    Args:
        - nk (int): Number of frequencies in spectrum.
        - nth (int): Number of directions in spectrum.

    Returns:
        - neigh (int): add description.

    """
    # build list of neighbours for each point
    nspec = nk * nth
    neigh = [[] for _ in range(nspec)]
    # neigh = np.zeros((9, nspec), dtype="int32")

    for n in range(nspec):
        ith = n % nth
        ik = n // nth

        if ik > 0:  # ... point at the bottom
            neigh[n].append(n - nth)

        if ik < nk - 1:  # ... point at the top
            neigh[n].append(n + nth)

        if ith > 0:  # ... point at the left
            neigh[n].append(n - 1)
        else:  # ... with wrap.
            neigh[n].append(n - 1 + nth)

        if ith < nth - 1:  # ... point at the right
            neigh[n].append(n + 1)
        else:  # ... with wrap.
            neigh[n].append(n + 1 - nth)

        if ik > 0 and ith > 0:  # ... point at the bottom-left
            neigh[n].append(n - nth - 1)
        elif ik > 0 and ith == 0:  # ... with wrap.
            neigh[n].append(n - nth - 1 + nth)

        if ik < nk - 1 and ith > 0:  # ... point at the top-left
            neigh[n].append(n + nth - 1)
        elif ik < nk - 1 and ith == 0:  # ... with wrap
            neigh[n].append(n + nth - 1 + nth)

        if ik > 0 and ith < nth - 1:  # ... point at the bottom-right
            neigh[n].append(n - nth + 1)
        elif ik > 0 and ith == nth - 1:  # ... with wrap
            neigh[n].append(n - nth + 1 - nth)

        if ik < nk - 1 and ith < nth - 1:  # ... point at the top-right
            neigh[n].append(n + nth + 1)
        elif ik < nk - 1 and ith == nth - 1:  # ... with wrap
            neigh[n].append(n + nth + 1 - nth)

        neigh[n] = neigh[n] + [-10 for nmiss in range(8 - len(neigh[n]))]

    return np.array(neigh)


def ptsort(imi, ihmax):
    """Sort discretised image."""
    numv = np.zeros(ihmax, dtype="int32")
    iaddr = np.zeros(ihmax,dtype="int32")
    nspec = len(imi)
    ind = np.zeros(nspec, dtype="int32")
    iorder = np.zeros(nspec, dtype="int32")

    # 1.  occurences per height
    for i in range(nspec):
        numv[imi[i]] = numv[imi[i]] + 1

    # 2.  starting address per height
    iaddr[0] = 0
    for i in range(ihmax-1):
        iaddr[i + 1] = iaddr[i] + numv[i]

    # 3.  order points
    for i in range(nspec):
        iv = imi[i]
        inn = iaddr[iv]
        iorder[i] = inn
        iaddr[iv] = inn + 1

    # 4.  sort points
    for i in range(nspec):
        ind[iorder[i]] = i

    return ind


def specpart(spectrum, neigh, nspec, ihmax, zp, imi):
    """Watershed partitioning.

    Args:
        - spectrum (2darray): Spectrum array E(f, d).
        - ihmax (int): Number of iterations.

    Returns:
        - part_array (2darray): Numbered partitions array with same shape as spectrum.

    """
    #  0.  initializations
    mstart = 0
    ic_label = 0
    ifict_pixel = -100
    iq1 = []

    neigh = [[v for v in list(neigh[i]) if v != -10] for i in range(len(neigh))]

    ind = zp.argsort()

    imo = -np.ones(nspec, dtype=int)
    imd = np.zeros(nspec, dtype=int)

    # 1.  loop over levels
    for ih in range(ihmax):
        # 1.a pixels at level ih
        for m in range(mstart, nspec):
            ip = ind[m]
            if imi[ip] != ih:
                break

            # flag the point, if it stays flagged, it is a separate minimum.
            imo[ip] = -2

            # if there is neighbor, set distance and add to queue.
            if any(imo[neigh[ip]] >= 0):
                imd[ip] = 1
                iq1.append(ip)

        # 1.b process the queue
        ic_dist = 1
        iq1.append(ifict_pixel)

        while True:
            ip = iq1.pop(0)

            # check for end of processing
            if ip == ifict_pixel:
                if not iq1:
                    break

                iq1.append(ifict_pixel)
                ic_dist += 1
                ip = iq1.pop(0)

            # process queue
            for ipp in neigh[ip]:
                # check for labeled watersheds or basins
                if imo[ipp] >= 0 and imd[ipp] < ic_dist:
                    if imo[ipp] > 0:
                        if imo[ip] in [-2, 0]:
                            imo[ip] = imo[ipp]
                        elif imo[ip] != imo[ipp]:
                            imo[ip] = 0
                    elif imo[ip] == -2:
                        imo[ip] = 0
                elif imo[ipp] == -2 and imd[ipp] == 0:
                    imd[ipp] = ic_dist + 1
                    iq1.append(ipp)

        # 1.c check for mask values in imo to identify new basins
        for ip in ind[mstart:m]:
            imd[ip] = 0
            if imo[ip] == -2:
                ic_label += 1  # ... new basin
                iq2 = [ip]
                while iq2:
                    imo[iq2] = ic_label
                    iqset = set([n for i in iq2 for n in neigh[i]])
                    # ... all masked points connected to it
                    iq2 = [i for i in iqset if imo[i] == -2]
        mstart = m

    # 2.  find nearest neighbor of 0 watershed points and replace
    #     use original input to check which group to affiliate with 0
    #     storing changes first in imd to assure symetry in adjustment.
    for _ in range(5):
        watershed0 = np.where(imo == 0)[0]
        if not any(watershed0):
            break

        newvals = []
        for jl in watershed0:
            jnl = [j for j in neigh[jl] if imo[j] != 0]
            if any(jnl):
                ipt = abs(zp[jnl] - zp[jl]).argmin()
                newvals.append(imo[jnl[ipt]])
            else:
                newvals.append(0)
        imo[watershed0] = newvals

    part_array = imo.reshape(spectrum.shape)
    return part_array


def nppart(spectrum, neigh, nspec, ihmax, zp, imi,   freq, dir, wspd, wdir, dpt, swells=3, agefac=1.7, wscut=0.3333):
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
    part_array = specpart(spectrum, neigh, nspec, ihmax, zp, imi)

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

    # Neighbours mapping
    neigh = ptnghb(dset.freq.size, dset.dir.size)

    # Input data for partitions to speed up computations
    nspec = dset.freq.size * dset.dir.size
    ihmax = 200
    zmin = dset.min(["freq", "dir"])
    zmax = dset.max(["freq", "dir"])
    zp = zmax - dset.stack({"fd": ("freq", "dir")})
    fact = (ihmax - 1) / (zmax - zmin)
    imi = (zp * fact).round().astype(int)
    # ind = zp.argsort()
    # imo = -np.ones(nspec, dtype=int)
    # imd = np.zeros(nspec, dtype=int)
    # imo = xr.full_like(zp, -1, dtype=int)
    # imd = xr.zeros_like(zp, dtype=int)

    # Partitioning full spectra
    dsout = xr.apply_ufunc(
        nppart,
        dset,
        neigh,
        nspec,

        ihmax,
        zp,
        imi,

        dset.freq,
        dset.dir,
        wspd,
        wdir,
        dpt,
        swells,
        agefac,
        wscut,
        input_core_dims=[["freq", "dir"], ["dummy1","dummy2"], [], [], ["fd"], ["fd"], ["freq"], ["dir"], [], [], [], [], [], []],
        # input_core_dims=[["freq", "dir"], ["dummy","fd"], [], [], [], [] ["freq"], ["dir"], [], [], [], [], [], []],
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


if __name__ == "__main__":
    from wavespectra import read_wavespectra

    dset = read_wavespectra("/source/consultancy/jogchum/route/route_feb21/p04/spec.nc")
    dsi = dset.isel(time=0, freq=slice(None, 10), dir=slice(None, 9)).load()
    spectrum = dsi.efth.values

    dsp = partition(dset.isel(time=slice(None, 1000))).load()
