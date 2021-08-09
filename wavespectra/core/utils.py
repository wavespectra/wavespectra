"""Miscellaneous functions."""
import copy
import datetime
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import griddata

from wavespectra.core.attributes import attrs


GAMMA = (
    lambda x: np.sqrt(2.0 * np.pi / x)
    * ((x / np.exp(1)) * np.sqrt(x * np.sinh(1.0 / x))) ** x
)
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def wavelen(freq, depth=None):
    """Wavelength L.

    Args:
        - freq (ndarray): Frequencies (Hz) for calculating L.
        - depth (float): Water depth, use deep water approximation by default.

    Returns;
        - L: ndarray of same shape as freq with wavelength for each frequency.

    """
    if depth is not None:
        ang_freq = 2 * np.pi * freq
        return 2 * np.pi / wavenuma(ang_freq, depth)
    else:
        return 1.56 / freq ** 2


def wavenuma(ang_freq, water_depth):
    """Chen and Thomson wavenumber approximation."""
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
        - C: ndarray of same shape as freq with wave celerity for each frequency.

    """
    if depth is not None:
        ang_freq = 2 * np.pi * freq
        return ang_freq / wavenuma(ang_freq, depth)
    else:
        return 1.56 / freq


def dnum_to_datetime(dnum):
    """Convert from numeric date to datetime."""
    return datetime.datetime.fromordinal(int(dnum) - 366) + datetime.timedelta(
        days=dnum % 1
    )


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


def to_datetime(np64):
    """Convert Datetime64 date to datatime."""
    if isinstance(np64, np.datetime64):
        dt = pd.to_datetime(str(np64)).to_pydatetime()
    elif isinstance(np64, xr.DataArray):
        dt = pd.to_datetime(str(np64.values)).to_pydatetime()
    else:
        OSError(
            "Cannot convert %s into datetime, expected np.datetime64 or xr.DataArray"
            % type(np64)
        )
    return dt



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
    ang_rot = 180 if coming_from else 0
    vetor = u + v * 1j
    mag = np.abs(vetor)
    direc = xr.ufuncs.angle(vetor, deg=True) + ang_rot
    direc = np.mod(90 - direc, 360)
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
    outfreq = infreq if outfreq is None or outfreq is False else outfreq
    outdir = indir if outdir is None or outdir is False else outdir

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
        dirs = D2R * (270 - np.expand_dims(outdir, 0))
        dirs2 = D2R * (270 - np.expand_dims(indir, 0))
        cosmat = np.dot(np.expand_dims(outfreq, -1), np.cos(dirs))
        sinmat = np.dot(np.expand_dims(outfreq, -1), np.sin(dirs))
        cosmat2 = np.dot(np.expand_dims(infreq, -1), np.cos(dirs2))
        sinmat2 = np.dot(np.expand_dims(infreq, -1), np.sin(dirs2))
        outspec = griddata(
            (cosmat2.flat, sinmat2.flat), inspec.flat, (cosmat, sinmat), method, 0.0
        )
    return outspec


def flatten_list(l, a):
    """Flatten list of lists"""
    for i in l:
        if isinstance(i, list):
            flatten_list(i, a)
        else:
            a.append(i)
    return a


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
