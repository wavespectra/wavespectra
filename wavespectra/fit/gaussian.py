"""Gaussian spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import pi

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates, to_coords
from wavespectra.core.attributes import attrs


def fit_gaussian(freq, hs, fp, gw=None, tm01=None, tm02=None, **kwargs):
    """Gaussian frequency spectrum (Bunney et al., 2014).

    Args:
        - freq (DataArray): Frequency array (Hz).
        - hs (DataArray, float): Significant wave height (m).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - gw (DataArray, float): Gaussian width parameter.
        - tm01 (DataArray, float): Mean wave period Tm.
        - tm02 (DataArray, float): Zero-upcrossing wave period Tz.

    Returns:
        - efth (SpecArray): Gaussian frequency spectrum E(f) (m2s).

    Note:
        - `tm01` and `tm02` are used to calculate the Gaussian width parameter `gw`
          when `gw` is not provided, they are ignored if `gw` is also specified.
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(hs, fp, gw, tm01, tm02)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    mo = (hs / 4) ** 2
    if gw is None:
        if tm02 is not None and tm01 is not None:
            sigma = np.sqrt((mo / tm02 ** 2) - (mo ** 2 / tm01 ** 2))
        else:
            raise ValueError("Either provide gw or tm01 and tm02 to calculate it")
    else:
        sigma = gw
        if tm01 is not None or tm02 is not None:
            logger.debug("gw has been provided, tm01 and tm02 are ignored")
    term1 = mo / (sigma * np.sqrt(2 * pi))
    term2 = np.exp(-((2 * pi * freq - 2 * pi * fp) ** 2 / (2 * (sigma) ** 2)))
    dsout = term1 * term2

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout
