"""Utility functions."""
import copy
import itertools
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from importlib import import_module
from inspect import getmembers, isfunction
from scipy.interpolate import griddata
from scipy.constants import pi, g

from wavespectra.core.attributes import attrs, set_spec_attributes


GAMMA = (
    lambda x: np.sqrt(2.0 * np.pi / x)
    * ((x / np.exp(1)) * np.sqrt(x * np.sinh(1.0 / x))) ** x
)
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def angle(dir1, dir2):
    """Relative angle between two directions.

    Args:
        - dir1 (array): First direction (degree).
        - dir2 (array): Second direction (degree).

    Returns:
        - angle (array): Angle difference between dir1 and dir2 (degree).

    """
    dif = np.absolute(dir1 % 360 - dir2 % 360)
    return np.minimum(dif, 360 - dif)


def waveage(dset, wspd, wdir, dpt, agefac):
    """Wave age criterion for partitioning wind-sea.

    Args:
        - wspd (xr.DataArray): Wind speed.
        - wdir (xr.DataArray): Wind direction.
        - dpt (xr.DataArray): Water depth.
        - agefac (float): Age factor.

    """
    wind_speed_component = agefac * wspd * np.cos(D2R * (dset.dir - wdir))
    wave_celerity = celerity(dset.freq, dpt)
    return wave_celerity <= wind_speed_component


def wavelen(freq, depth=None):
    """Wavelength L.

    Args:
        - freq (ndarray): Frequencies (Hz) for calculating L.
        - depth (float): Water depth, use deep water approximation by default.

    Returns;
        - L: ndarray of same shape as freq with wavelength for each frequency.

    """
    if depth is not None:
        return 2 * np.pi / wavenuma(freq, depth)
    else:
        return 1.56 / freq ** 2


def wavenuma(freq, water_depth):
    """Chen and Thomson wavenumber approximation.

    Args:
        freq (DataArray, 1darray, float): Frequencies (Hz).
        water_depth (DataArray, float): Water depth (m).

    Returns:
        k (DataArray, 1darray, float): Wavenumber 2pi / L.

    """
    ang_freq = 2 * np.pi * freq
    k0h = 0.10194 * ang_freq * ang_freq * water_depth
    D = [0, 0.6522, 0.4622, 0, 0.0864, 0.0675]
    a = 1.0
    for i in range(1, 6):
        a += D[i] * k0h ** i
    return (k0h * (1 + 1.0 / (k0h * a)) ** 0.5) / water_depth


def celerity(freq, depth=None):
    """Wave celerity C.

    Args:
        - freq (ndarray): Frequencies (Hz) for calculating C.
        - depth (float): Water depth, use deep water approximation by default.

    Returns;
        - C: ndarray of same shape as freq with wave celerity (m/s) for each frequency.

    """
    if depth is not None:
        ang_freq = 2 * np.pi * freq
        return ang_freq / wavenuma(freq, depth)
    else:
        return 1.56 / freq


def to_nautical(ang):
    """Convert from cartesian to nautical angle."""
    return np.mod(270 - ang, 360)


def unique_indices(ds, dim="time"):
    """Remove duplicate indices from dataset.

    Args:
        - ds (Dataset, DataArray): Dataset to remove duplicate indices from.
        - dim (str): Dimension to remove duplicate indices from.

    Returns:
        dsout (Dataset, DataArray): Dataset with duplicate indices along dim removed.

    """
    _, index = np.unique(ds[dim], return_index=True)
    return ds.isel(**{dim: index})


def unique_times(ds):
    """Remove duplicate times from dataset."""
    return unique_indices(ds, "time")


def spddir_to_uv(spd, direc, coming_from=False):
    """Converts (spd, dir) to (u, v).

    Args:
        spd (array): magnitudes to convert.
        direc (array): directions to convert (degree).
        coming_from (bool): True if directions in coming-from convention,
            False if in going-to.

    Returns:
        u (array): eastward wind component.
        v (array): northward wind component.

    """
    ang_rot = 180 if coming_from else 0
    direcR = np.deg2rad(direc + ang_rot)
    u = spd * np.sin(direcR)
    v = spd * np.cos(direcR)
    return u, v


def uv_to_spddir(u, v, coming_from=False):
    """Converts (u, v) to (spd, dir).

    Args:
        u (array): eastward wind component.
        v (array): northward wind component.
        coming_from (bool): True for output directions in coming-from convention,
            False for going-to.

    Returns:
        mag (array): magnitudes.
        direc (array): directions (degree).

    """
    to_nautical = 270 if coming_from else 90
    mag = np.sqrt(u ** 2 + v ** 2)
    direc = np.rad2deg(np.arctan2(v, u))
    direc = (to_nautical - direc) % 360
    return mag, direc


def interp_spec(inspec, infreq, indir, outfreq=None, outdir=None, method="linear"):
    """Interpolate onto new spectral basis.

    Args:
        inspec (2D ndarray): input spectrum E(infreq,indir) to be interpolated.
        infreq (1D ndarray): frequencies of input spectrum.
        indir (1D ndarray): directions of input spectrum.
        outfreq (1D ndarray): frequencies of output interpolated spectrum, same as
            infreq by default.
        outdir (1D ndarray): directions of output interpolated spectrum, same as
            infreq by default.
        method: {'linear', 'nearest', 'cubic'}, method of interpolation to use with
            griddata.

    Returns:
        outspec (2D ndarray): interpolated ouput spectrum E(outfreq,outdir).

    Note:
        If either outfreq or outdir is None or False this coordinate is not interpolated
        Choose indir=None if spectrum is 1D.

    TODO: Deprecate in favour of new regrid_spec function.

    """
    ndim = inspec.ndim
    if ndim > 2:
        raise ValueError(f"interp_spec requires 2d spectra but inspec has {ndim} dims")

    if outfreq is None:
        outfreq = infreq
    if outdir is None:
        outdir = indir

    if (np.array_equal(infreq, outfreq)) & (np.array_equal(indir, outdir)):
        outspec = copy.deepcopy(inspec)
    elif np.array_equal(indir, outdir):
        if indir is not None:
            outspec = np.zeros((len(outfreq), len(outdir)))
            for idir in range(len(indir)):
                outspec[:, idir] = np.interp(
                    outfreq, infreq, inspec[:, idir], left=0.0, right=0.0
                )
        else:
            outspec = np.interp(
                outfreq, infreq, np.array(inspec).ravel(), left=0.0, right=0.0
            )
    else:
        outdir = D2R * (270 - np.expand_dims(outdir, 0))
        outcos = np.dot(np.expand_dims(outfreq, -1), np.cos(outdir))
        outsin = np.dot(np.expand_dims(outfreq, -1), np.sin(outdir))
        indir = D2R * (270 - np.expand_dims(indir, 0))
        incos = np.dot(np.expand_dims(infreq, -1), np.cos(indir)).flat
        insin = np.dot(np.expand_dims(infreq, -1), np.sin(indir)).flat
        outspec = griddata((incos, insin), inspec.flat, (outcos, outsin), method, 0.0)
    return outspec


def flatten_list(l, a):
    """Flatten list of lists"""
    for i in l:
        if isinstance(i, list):
            flatten_list(i, a)
        else:
            a.append(i)
    return a


def scaled(spec, hs):
    """Scale spectra.

    The energy density in each spectrum is scaled by a single factor so that
        significant wave height calculated from the scaled spectrum corresponds to hs.

    Args:
        - spec (SpecArray, SpecDataset): Wavespectra object to be scaled.
        - hs (DataArray, float): Hs values to use for scaling, if float it will scale
          each spectrum in the dataset, if a DataArray it must have all non-spectral
          coordinates as the spectra dataset.

    Returns:
        - spec (SpecArray, SpecDataset): Scaled wavespectra object.

    """
    fac = (hs / spec.spec.hs()) ** 2
    return fac * spec


def check_same_coordinates(*args):
    """Check if DataArrays have same coordinates."""
    for darr1, darr2 in itertools.combinations(args, 2):
        if isinstance(darr1, xr.DataArray) and isinstance(darr2, xr.DataArray):
            if not darr1.coords.to_dataset().equals(darr2.coords.to_dataset()):
                raise ValueError(f"{darr1.name} and {darr2.name} must have same coords")
        elif isinstance(darr1, xr.Dataset) or isinstance(darr2, xr.Dataset):
            raise TypeError(
                f"Only DataArrays should be compared, got {type(darr1)}, {type(darr2)}"
            )


def load_function(module_name, func_name, prefix=None):
    """Returns a function object from string.

    Args:
        - module_name (str): Name of module to import function from.
        - func_name (str): Name of function to import.
        - prefix (str): Used to filter available functions in exception.

    """
    module = import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        members = getmembers(module, isfunction)
        if prefix is not None:
            # Check for functions starting with prefix
            funcs = [mem[0] for mem in members if mem[0].startswith(prefix)]
        else:
            # Check for functions defined in module (exclude those imported in module)
            funcs = [mem[0] for mem in members if mem[1].__module__ == module.__name__]
        raise AttributeError(
            f"'{func_name}' not available in {module.__name__}, available are: {funcs}"
        ) from exc


def to_coords(array, name):
    """Create coordinates DataArray.

    Args:
        - array (list, 1darray): Coordinate values.
        - name (str): Coordinate name.

    Returns:
        coords (DataArray): Coordinates DataArray.

    """
    coords = xr.DataArray(array, coords={name: array}, dims=(name,))
    set_spec_attributes(coords)
    return coords


def regrid_spec(dset, freq=None, dir=None, maintain_m0=True):
    """Regrid spectra onto new spectral basis.

    Args:
        - dset (Dataset, DataArray): Spectra to interpolate.
        - freq (DataArray, 1darray): Frequencies of interpolated spectra (Hz).
        - dir (DataArray, 1darray): Directions of interpolated spectra (deg).
        - maintain_m0 (bool): Ensure variance is conserved in interpolated spectra.

    Returns:
        - dsi (Dataset, DataArray): Regridded spectra.

    Note:
        - All freq below lowest freq are interpolated assuming :math:`E_d(f=0)=0`.
        - :math:`Ed(f)` is set to zero for new freq above the highest freq in dset.
        - Only the 'linear' method is currently supported.
        - Duplicate wrapped directions (e.g., 0 and 360) are removed when regridding
          directions because indices must be unique to intepolate.

    """
    dsout = dset.copy()
    if isinstance(freq, (list, tuple)):
        freq = np.array(freq)
    if isinstance(dir, (list, tuple)):
        dir = np.array(dir)

    if dir is not None:
        dsout = dsout.assign_coords({attrs.DIRNAME: dsout[attrs.DIRNAME] % 360})

        # Remove any duplicate direction index
        dsout = unique_indices(dsout, attrs.DIRNAME)

        # Interpolate heading
        dsout = dsout.sortby('dir')
        to_concat = [dsout]

        # Repeat the first and last direction with 360 deg offset when required
        if dir.min() < dsout.dir.min():
            highest = dsout.isel(dir=-1)
            highest['dir'] = highest.dir - 360
            to_concat = [highest, dsout]
        if dir.max() > dsout.dir.max():
            lowest = dsout.isel(dir=0)
            lowest['dir'] = lowest.dir + 360
            to_concat.append(lowest)

        if len(to_concat) > 1:
            dsout = xr.concat(to_concat, dim='dir')

        # Interpolate directions
        dsout = dsout.interp(dir=dir, assume_sorted=True)

    if freq is not None:
        # If needed, add a new frequency at f=0 with zero energy
        if freq.min() < dsout.freq.min():
            fzero = 0 * dsout.isel(freq=0)
            fzero['freq'] = 0
            dsout = xr.concat([fzero, dsout], dim='freq')

        # Interpolate frequencies
        dsout = dsout.interp(freq=freq, assume_sorted=False, kwargs={'fill_value': 0})

    if maintain_m0:
        scale = dset.spec.hs()**2 / dsout.spec.hs()**2
        dsout = dsout * scale

    return dsout


def smooth_spec(dset, freq_window=3, dir_window=3):
    """Smooth spectra with a running average.

    Args:
        - dset (Dataset, DataArray): Spectra to smooth.
        - freq_window (int): Rolling window size along `freq` dim.
        - dir_window (int): Rolling window size along `dir` dim.

    Returns:
        - efth (DataArray): Smoothed spectra.

    Note:
        - The window size should be an odd value to ensure symmetry.

    """
    for window in [freq_window, dir_window]:
        if (window % 2) == 0:
            raise ValueError(
                f"Window size must be an odd value to ensure symmetry, got {window}"
            )

    dsout = dset.sortby(attrs.DIRNAME)

    # Avoid problems when extending dirs with wrong data type
    dsout[attrs.DIRNAME] = dset[attrs.DIRNAME].astype("float32")

    # Extend circular directions to take care of edge effects
    dirs = dsout[attrs.DIRNAME].values
    dd = list(set(np.diff(dirs)))
    if len(dd) == 1:
        dd = float(dd[0])
        is_circular = (abs(dirs.max() - dirs.min() + dd - 360)) < (0.1 * dd)
    else:
        is_circular = False
    if is_circular:
        # Extend directions on both sides
        left = dsout.isel(**{attrs.DIRNAME: slice(-window, None)})
        left = left.assign_coords({attrs.DIRNAME: left[attrs.DIRNAME] - 360})
        right = dsout.isel(**{attrs.DIRNAME: slice(0, window)})
        right = right.assign_coords({attrs.DIRNAME: right[attrs.DIRNAME] + 360})
        dsout = xr.concat([left, dsout, right], dim=attrs.DIRNAME)

    # Smooth
    dim = {attrs.FREQNAME: freq_window, attrs.DIRNAME: dir_window}
    dsout = dsout.rolling(dim=dim, center=True).mean()

    # Clip to original shape
    if not dsout[attrs.DIRNAME].equals(dset[attrs.DIRNAME]):
        dsout = dsout.sel(**{attrs.DIRNAME: dset[attrs.DIRNAME]})
        dsout = dsout.chunk(**{attrs.DIRNAME: -1})

    # Fill missing values at boundaries using original spectra
    dsout = xr.where(dsout.notnull(), dsout, dset)

    return dsout.assign_coords(dset.coords)


def is_overlap(rect1, rect2):
    """Check if rectangles overlap.

    Args:
        - rect1 (list): Bounding box of the 1st rectangle [l1, b1, r1, t1].
        - rect2 (list): Bounding box of the 2nd rectangle [l2, b2, r2, t2].

    Returns:
        - True if the two rectangles overlap, False otherwise.

    """
    l1, b1, r1, t1 = rect1
    l2, b2, r2, t2 = rect2
    if (r1 <= l2) or (r2 <= l1):
        return False
    if (t1 <= b2) or (t2 <= b1):
        return False
    return True


def dfp_wsea(wspd: xr.DataArray, fp: xr.DataArray, dt: float, scaling: float = 1.0) -> xr.DataArray:
    """Rate of change of the wind-sea peak wave frequency.
    Based on fetch-limited relationships, (Ewans & Kibblewhite, 1986).

    Ewans, K.C., and A.C. Kibblewhite (1988).
    Spectral characteristics of ocean waves undergoing generation in New Zealand waters.
    9th Australian Fluid Mechanics Conference, Auckland, 8-12 December, 1986.

    Parameters
    ----------
    wspd: xr.DataArray
        Wind speed (m/s)
    t: float
        Time duration (s)
    scaling: float
        Scaling parameter

    """

    tmp = (15.8*(g/wspd)**0.57)

    t0 = ( fp / tmp ) ** (-1/0.43)
    return scaling * tmp * (t0+dt)**(-0.43) - fp


def dfp_swell(dt: xr.DataArray, distance: xr.DataArray = 1e6) -> xr.DataArray:
    """Rate of change of the swell peak wave frequency.
    Based on the swell dispersion relationship derived by Snodgrass et al (1966).

    Snodgrass, F. E., G. W. Groves, K. F. Hasselmann, G. R. Miller, and W. H. Munk (1966).
    Propagation of Ocean Swell across the Pacific.
    Philosophical Transactions of the Royal Society of London.
    Series A, Mathematical and Physical Sciences, Vol. 259, No. 1103 (May 5, 1966), pp. 431-497

    Parameters
    ----------
    dt: xr.DataArray
        Time difference
    distance: xr.DataArray
        Distance to the swell source (m)

    """
    return dt * g / (4 * pi * distance)


def partition_distance(curr_stats,
                       prev_stats,
                       dfp_sea_max,
                       dfp_swell_max,
                       ddpm_sea_max,
                       ddpm_swell_max,
                       idt=1):
    """Calculate the distance between two partitions based
    on normalized peak frequency and mean peak direction deltas.

    Parameters
    ----------
    curr_stats: xr.Dataset
        Statistics of the current partition
    prev_stats: xr.Dataset
        Statistics of the previous partition
    dfp_sea_max: xr.DataArray
        Maximum delta fp for sea partitions
    dfp_swell_max: float
        Maximum delta fp for swell partitions
    ddpm_sea_max: float
        Maximum delta dpm for sea partitions
    ddpm_swell_max: float
        Maximum delta dpm for swell partitions
    idt: int
        Number of time steps between current and previous partition

    """

    # Caculate angle difference between current and previous partition
    ddpm = np.abs(((curr_stats.dpm.values-prev_stats.dpm.values)+180)%360-180)

    # Pick angle threshold based on partition type
    ddpm_max = ddpm_sea_max if prev_stats.part.values == 0 else ddpm_swell_max
    # If angle threshold is exceeded return 999, else carry on
    if ddpm > ddpm_max:
        return 999

    # Calculate peak frequency difference between current and previous partition
    dfp = curr_stats.fp.values-prev_stats.fp.values

    # If the partition was a sea partition we expect a negative delta
    # hence we use the swell partition maximum delta as a maximum threshold
    dfp_max = dfp_swell_max*idt
    if dfp > dfp_max:
        return 999

    # Minimum threshold is based on the sea/swell partition maximum delta
    #  depending on the partition type
    if prev_stats.part.values == 0: # Sea partition we expect a negative delta
        dfp_min = dfp_sea_max*idt
    else: # Swell partition we expect a positive delta
        dfp_min = -dfp_swell_max*idt
    if dfp < dfp_min:
        return 999

    # Return distance between current and previous partition
    # as a weighted sum of normalized dfp and ddpm
    return (np.abs(dfp)/max(dfp_max, np.abs(dfp_min))*idt + ddpm/ddpm_max)


def match_consecutive_partitions(stats,
                                 dfp_sea_max,
                                 dfp_swell_max,
                                 ddpm_sea_max,
                                 ddpm_swell_max):
    """
    Match partitions of consecutive spectra based on evolution of peak frequency
    and peak direction.

    Parameters
    ----------
    stats: xr.Dataset
        Statistics of the spectral partitions.
    dfp_sea_max: float
        Maximum delta fp for sea partitions.
    dfp_swell_max: float
        Maximum delta fp for swell partitions.
    ddpm_sea_max: float
        Maximum delta dpm for sea partitions.
    ddpm_swell_max: float
        Maximum delta dpm for swell partitions.

    Returns
    -------
    matches: np.ndarray
        Array of matches between partitions of consecutive spectra.
        Array contains value in nth position contains the partition number in
        the previous time step that matches the partition number n in the
        current time step.
        -999 is for nans and -888 is for partitions that have no match.

    """

    # Initialise matches to -999
    matches = np.ones_like(stats.part.values, dtype='int16')*-999

    # Those are all the partitions in the previous time step that have energy
    available =\
        [ip for ip, fp_is_not_nan in enumerate(~np.isnan(stats.isel(time=0).fp.values)) if fp_is_not_nan]

    # Loop over all partitions in the current time step
    for ip_curr in range(stats.part.size):

        if ~np.isnan(stats.isel(part=ip_curr, time=1).fp.values):

            # Find all possible matches for the current partition sorted by increasing distance
            part_matches =\
                [v[0] for v in sorted([(ip_prev,
                                        partition_distance(curr_stats=stats.isel(time=1,
                                                                                 part=ip_curr),
                                                           prev_stats=stats.isel(time=0,
                                                                                 part=ip_prev),
                                                           dfp_sea_max=dfp_sea_max,
                                                           dfp_swell_max=dfp_swell_max,
                                                           ddpm_sea_max=ddpm_sea_max,
                                                           ddpm_swell_max=ddpm_swell_max))
                                       for ip_prev in available],
                                      key=lambda x: x[-1])
                            if v[-1] != 999]

            # If no match found, create new partition to track
            if len(part_matches) == 0:
                matches[ip_curr] = -888
            else: # If match found mark it and remove it from the available list
                matches[ip_curr] = part_matches[0]
                available.remove(part_matches[0])

    return matches


def track_partitions(stats,
                     wspd,
                     ddpm_sea_max=30,
                     ddpm_swell_max=20,
                     dfp_sea_scaling=1,
                     dfp_swell_source_distance=1):
    """
    Track partitions in a series of consecutive spectra based on
    the evolution of peak frequency and peak direction.
    Partitions are matched with the closest partition in the frequency-direction
    space of the previous time step for which the difference in direction with less
    than ddpm_max and the difference in peak frequency is less than dfp_max.
    ddpm_max differs for sea and swell partitions and is set manually.
    dfp_max also differs for sea and swell partitions. In the case of sea partitions
    it is a function of wind speed and is set to the rate of change of the wind-sea peak
    wave frequency estimated from fetch-limited relationships, (Ewans & Kibblewhite, 1986).
    In the case of swell partitions it is set to the rate of change of the swell peak wave
    frequency based on the swell dispersion relationship derived by Snodgrass et al (1966)
    assuming the distance to the source is 1e6 m.

    Parameters
    ----------
    stats: xr.Dataset
        Statistics of the spectral partitions.
    wspd: xr.DataArray
        Wind speed (m/s).
    ddpm_sea_max: float
        Maximum delta dpm for sea partitions.
        Default is 30 degrees.
    ddpm_swell_max: float
        Maximum delta dpm for swell partitions.
        Default is 20 degrees.
    dfp_sea_scaling: float
        Scaling parameter for the rate of change of the wind-sea peak wave frequency.
        Default is 1.
    dfp_swell_source_distance: float
        Distance to the swell source (m) for the rate of change of the swell peak wave
        frequency. Default is 1e6 m.

    Returns
    -------
    part_ids: np.ndarray
        Array containing the partition ids for each partition and each time step.
        -999 is for nans.
    n_parts: int
        Number of partitions tracked.

    """

    # Assuming that the time step is constant across the dataset
    # calculate dt in seconds
    dt = float(stats.time[:2].diff("time") / np.timedelta64(1, 's') )

    # Calculate maximum delta fp for sea partitions
    # as it is a function of wind speed this is a data array
    dfp_sea_max = dfp_wsea(wspd=wspd,
                           fp=stats.isel(part=0).fp,
                           dt=dt,
                           scaling=1.)

    # Calculate maximum delta fp for swell partitions
    # it is a scalar
    dfp_swell_max = dfp_swell(dt=dt,
                             distance=1e6)


    # Calculate local matches (match partitions between consecutive time steps)
    # -999 is for nans and -888 is for partitions that have no match
    part_ids =\
        np.hstack([np.ones((stats.part.size,1), dtype='int16')*-999]\
                  +[match_consecutive_partitions(stats.isel(time=slice(it-1, it+1)).copy(),
                                                 dfp_sea_max.isel(time=it-1).values,
                                                 dfp_swell_max,
                                                 ddpm_sea_max,
                                                 ddpm_swell_max).reshape((-1,1))
                   for it in range(1, stats.time.size)])


    # Turn the local matches into global matches
    part_id = 0 # Partitions are numbered from 0

    # Number the partitions in the first time step
    for ip, fp in enumerate(stats.isel(time=0).fp.values):
        if ~np.isnan(fp):
            part_ids[ip, 0] = part_id
            part_id += 1

    # Propagate the partition ids through time
    for it in range(1, stats.time.size):
        for ip in range(stats.part.size):
            if part_ids[ip, it] == -888:
                part_ids[ip, it] = part_id
                part_id += 1
            elif part_ids[ip, it] != -999:
                part_ids[ip, it] = part_ids[part_ids[ip, it], it-1]

    return part_ids, part_id