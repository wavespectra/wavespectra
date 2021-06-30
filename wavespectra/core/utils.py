"""Utility functions."""
import copy
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from importlib import import_module
from inspect import getmembers, isfunction
from scipy.interpolate import griddata


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


def unique_times(ds):
    """Remove duplicate times from dataset."""
    _, index = np.unique(ds["time"], return_index=True)
    return ds.isel(time=index)


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
    for darr1, darr2 in zip(args[:-1], args[1:]):
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
