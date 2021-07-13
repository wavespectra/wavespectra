import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import D2R, R2D, wavenuma, scaled, check_same_coordinates
from wavespectra.core.attributes import attrs, set_spec_attributes


def cartwright(dir, dm, dspr, under_90=False):
    """Cosine-squared directional spreading of Cartwright (1963).

    :math:`G(\\theta,f)=F(s)cos^{2}\\frac{1}{2}(\\theta-\\theta_{m})`

    Args:
        - dir (DataArray): Wave directions (degree).
        - dm: (DataArray, float): Mean wave direction (degree).
        - dspr (DataArray, float): Directional spreading (degree).
        - under_90 (bool): Zero the spreading curve above 90 degrees range from dm.

    Returns:
        - gth (DataArray): Normalised spreading function :math:`g(\\theta)`.

    Note:
        - If `dm` and `dspr` are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(dm, dspr)

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


def bunney(dir, freq, dm, dpm, dspr, dpspr, fm, fp):
    """Asymmetrical directional spreading of Bunney et al. (2014).

    Args:
        - dir (DataArray): Wave directions (degree).
        - freq (DataArray): Wave frequency (Hz).
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

    # Convert to radians
    theta_p = D2R * dpm
    theta_m = D2R * dm
    sigma_p = D2R * dpspr
    sigma_m = D2R * dspr

    # Gradients
    dtheta = (theta_p - theta_m) / (fp - fm)
    dsigma = (sigma_p - sigma_m) / (fp - fm)

    # Modified peak direction
    theta = theta_p + dtheta * (freq - fp)
    theta = (R2D * theta.where(freq > fp, theta_p)) % 360

    # Modified spread parameter
    sigma = sigma_p + dsigma * (freq - fp)
    sigma = R2D * sigma.where(freq > fp, sigma_p)

    # Apply cosine-square to modified parameters
    gfth = cartwright(dir, theta, sigma, under_90=False)

    return gfth
