"""Utility functions."""
import copy
import itertools
import numpy as np
import xarray as xr
from importlib import import_module
from inspect import getmembers, isfunction
from scipy.interpolate import griddata

from wavespectra.core.attributes import attrs, set_spec_attributes


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


def waveage(freq, dir, wspd, wdir, dpt, agefac):
    """Wave age criterion for partitioning wind-sea.

    Args:
        - freq (xr.DataArray): Spectral frequencies.
        - dir (xr.DataArray): Spectral directions.
        - wspd (xr.DataArray): Wind speed.
        - wdir (xr.DataArray): Wind direction.
        - dpt (xr.DataArray): Water depth.
        - agefac (float): Age factor.

    """
    wind_speed_component = agefac * wspd * np.cos(D2R * (dir - wdir))
    wave_celerity = celerity(freq, dpt)
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
        return 1.56 / freq**2


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
        a += D[i] * k0h**i
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
    mag = np.sqrt(u**2 + v**2)
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


def flatten_list(list_to_flat, list_to_append_into):
    """Flatten list of lists"""
    for i in list_to_flat:
        if isinstance(i, list):
            flatten_list(i, list_to_append_into)
        else:
            list_to_append_into.append(i)
    return list_to_append_into


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
        dsout = dsout.sortby("dir")
        to_concat = [dsout]

        # Repeat the first and last direction with 360 deg offset when required
        if dir.min() < dsout.dir.min():
            highest = dsout.isel(dir=-1)
            highest["dir"] = highest.dir - 360
            to_concat = [highest, dsout]
        if dir.max() > dsout.dir.max():
            lowest = dsout.isel(dir=0)
            lowest["dir"] = lowest.dir + 360
            to_concat.append(lowest)

        if len(to_concat) > 1:
            dsout = xr.concat(to_concat, dim="dir")

        # Interpolate directions
        dsout = dsout.interp(dir=dir, assume_sorted=True)

    if freq is not None:
        # If needed, add a new frequency at f=0 with zero energy
        if freq.min() < dsout.freq.min():
            fzero = 0 * dsout.isel(freq=0)
            fzero["freq"] = 0
            dsout = xr.concat([fzero, dsout], dim="freq")

        # Interpolate frequencies
        dsout = dsout.interp(freq=freq, assume_sorted=False, kwargs={"fill_value": 0})

    if maintain_m0:
        scale = dset.spec.hs() ** 2 / dsout.spec.hs() ** 2
        dsout = dsout * scale

    if isinstance(dsout, xr.DataArray):
        dsout.name = "efth"
    set_spec_attributes(dsout)
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

    # Assign coords from original dataset
    dsout = dsout.assign_coords(dset.coords)

    # Fill missing values at boundaries using original spectra
    dsout = xr.where(dsout.notnull(), dsout, dset)

    set_spec_attributes(dsout)

    return dsout


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
