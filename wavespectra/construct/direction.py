import numpy as np
import xarray as xr
from scipy.constants import pi

from wavespectra import SpecArray
from wavespectra.core.utils import R2D, check_same_coordinates, to_coords
from wavespectra.core.attributes import attrs


def cartwright(dir, dm, dspr, under_90=False, **kwargs):
    """Cosine-squared directional spreading of Cartwright (1963).

    :math:`G(\\theta,f)=F(s)cos^{2}\\frac{1}{2}(\\theta-\\theta_{m})`

    Args:
        - dir (DataArray, 1darray, list): Wave direction coords (degree).
        - dm: (DataArray, float): Mean wave direction (degree).
        - dspr (DataArray, float): Directional spreading (degree).
        - under_90 (bool): Zero the spreading curve above 90 degrees range from dm.

    Returns:
        - gth (DataArray): Normalised spreading function :math:`g(\\theta)`.

    Note:
        - If `dm` and `dspr` are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(dm, dspr)
    if not isinstance(dir, xr.DataArray):
        dir = to_coords(dir, "dir")

    # Angular difference from mean direction:
    dth = np.abs(dir - dm)
    dth = dth.where(dth <= 180, 360.0 - dth)  # for directional wrapping

    # spread function:
    s = 2.0 / (np.deg2rad(dspr) ** 2) - 1
    gth = np.cos(0.5 * np.deg2rad(dth)) ** (2 * s)

    # mask directions +-90 deg from mean
    if under_90:
        gth = gth.where(np.abs(dth) <= 90.0, 0.0)

    # normalise
    gsum = 1.0 / (gth.sum(attrs.DIRNAME) * (2 * pi / dir.size))
    gth = gth * gsum

    return gth / R2D


def asymmetric(dir, freq, dm, dpm, dspr, dpspr, fm, fp, **kwargs):
    """Asymmetric directional spreading of Bunney et al. (2014).

    Args:
        - dir (DataArray, 1darray, list): Wave direction coords (degree).
        - freq (DataArray, 1darray, list): Wave frequency coords (Hz).
        - dm: (DataArray, float): Mean wave direction (degree).
        - dpm: (DataArray, float): Peak wave direction (degree).
        - dspr (DataArray, float) Mean directional spreading (degree).
        - dpspr (DataArray, float) Peak directional spreading (degree).
        - fm (DataArray, float) Mean wave frequency (Hz).
        - fp (DataArray, float) Peak wave frequency (Hz).

    Returns:
        - gfth (DataArray): Modified normalised spreading function :math:`g(f,\\theta)`.

    Note:
        - If arguments other than `dir` and `freq` are DataArrays
          they must share the same coordinates.

    """
    check_same_coordinates(dm, dpm, dspr, dpspr, fm, fp)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")
    if not isinstance(dir, xr.DataArray):
        dir = to_coords(dir, "dir")

    # Gradients
    # ==========
    # Limiters to avoid negative and large numbers
    dd = dm - dpm
    ds = np.maximum(dspr - dpspr, 0)
    df = np.maximum(fm - fp, 0.001)
    dddf = dd / df
    dsdf = ds / df

    # Modified peak direction
    # ========================
    # Limiter for frequency band peak directon (30% peak spread)
    theta = dpm + dddf * (freq - fp)
    theta = np.minimum(1.5 * dpm, np.maximum(0.5 * dpm, theta))

    # Modified spread parameter
    # ==========================
    # Limiter for frequency band spreading, 0.14 limits s to ~100 for narrow beamwidths
    sigma = dpspr + dsdf * (freq - fp)
    sigma = np.maximum(0.5 * dspr, np.minimum(1.5 * np.maximum(dspr, dpspr), sigma))
    sigma = sigma.where(sigma >= 0.14, 0.14)

    # Apply cosine-square to modified parameters
    # ===========================================
    gfth = cartwright(dir, theta, sigma, under_90=False)

    return gfth
