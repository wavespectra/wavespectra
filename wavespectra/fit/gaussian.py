"""Gaussian spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

import matplotlib.pyplot as plt

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates
from wavespectra.core.attributes import attrs


def gaussian(freq, hs, fp, tm01, tm02):
    """Gaussian frequency spectrum (Bunney et al., 2014).

    Args:
        - freq (DataArray): Frequency array (Hz).
        - hs (DataArray, float): Significant wave height (m).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - tm01 (DataArray, float): Mean wave period Tm.
        - tm02 (DataArray, float): Zero-upcrossing wave period Tz.

    Returns:
        - efth (SpecArray): Gaussian frequency spectrum E(f) (m2s).

    Note:
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    mo = (hs / 4) ** 2
    sigma = np.sqrt( (mo / tm02**2) - (mo**2 / tm01**2) )
    term1 = mo / (sigma * np.sqrt(2 * pi))
    term2 = np.exp( -((2 * pi * freq - 2 * pi * fp)**2 / (2 * (sigma)**2)) )
    dsout = term1 * term2

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout
