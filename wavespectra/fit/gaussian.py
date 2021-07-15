"""Gaussian spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import pi

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates, to_coords
from wavespectra.core.attributes import attrs


def fit_gaussian(freq, hs, fp, gw=None, **kwargs):
    """Gaussian frequency spectrum (Bunney et al., 2014).

    Args:
        - freq (DataArray): Frequency array (Hz).
        - hs (DataArray, float): Significant wave height (m).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - gw (DataArray, float): Gaussian width parameter :math:`\sigma`.

    Returns:
        - efth (SpecArray): Gaussian frequency spectrum E(f) (m2s).

    Note:
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(hs, fp, gw)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    mo = (hs / 4) ** 2
    term1 = mo / (gw * np.sqrt(2 * pi))
    term2 = np.exp(-((2 * pi * freq - 2 * pi * fp) ** 2 / (2 * (gw) ** 2)))
    dsout = term1 * term2

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout
