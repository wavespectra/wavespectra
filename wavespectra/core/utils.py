"""Miscellaneous functions."""
import copy
import datetime
import numpy as np
import pandas as pd
import xarray as xr

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


def unique_times(ds):
    """Remove duplicate times from dataset."""
    _, index = np.unique(ds["time"], return_index=True)
    return ds.isel(time=index)


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


def bins_from_frequency_grid(bin_centers, absolute_tolerance=1e-3):
    """Determines the location of the edges of bins from the provided bin centers.

    Detects the pre-coded common frequency grids:
    - exponential grids
    - datawell waverider grid


    constant grids:
    - delta between frequencies is constant

    exponential grids:
    - frequencies are at  c1 * exp(c2 * i); bin edges are at

    Args:
        bin_centers : iterable [Hz or rad/s]
        absolute_tolerance : maximum absolute deviation in frequency step

    Returns:
        left, right, width : bin edges and bin widths [Hz rad/s, same as input]

    """

    freqs = np.array(bin_centers, dtype=float)
    nfreqs = len(freqs)

    # Check if the frequency grid is constant
    width = np.diff(freqs)
    if (np.max(width) - np.min(width)) < absolute_tolerance:
        centers = np.linspace(min(freqs), max(freqs), nfreqs)
        left = centers - 0.5 * np.mean(width)
        right = centers + 0.5 * np.mean(width)
        width = right - left
        return left, right, width, centers

    # Grid is not constant.
    # Check it the frequency grid is exponential
    #
    # to determine the bin-size we need to re-construct the original bin boundaries
    #
    # c2 is the n-log of the frequency increase factor
    # take the average to minimize the effect of lack of significant digits

    # Check it the frequency grid is exponential
    # freq_center_i = c1 * exp(c2 * i)

    increase_factor = np.mean(freqs[1:] / freqs[:-1])
    c2 = np.log(increase_factor)

    # determine c1 from the highest bin
    c1 = freqs[-1] / np.exp(c2 * nfreqs)

    # check the assumptions
    ifreq = np.arange(1, nfreqs + 1)
    freq_check = c1 * np.exp(c2 * ifreq)

    difference = freqs - freq_check
    max_difference = np.max(np.abs(difference))

    if max_difference < absolute_tolerance:
        # we have an exponential grid

        left = c1 * np.exp(c2 * (ifreq - 0.5))
        right = c1 * np.exp(c2 * (ifreq + 0.5))
        width = right - left
        centers = freq_check

        return left, right, width, centers

    # grid is not exponential and not constant

    raise ValueError(
        "The type of frequency grid can not be detected. Clean input data or increase tolerance?"
    )
