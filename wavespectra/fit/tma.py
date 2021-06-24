"""TMA spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates, wavenuma
from wavespectra.core.attributes import attrs
from wavespectra.fit.jonswap import jonswap

def tma(freq, hs, tp, dep, alpha=0.0081, gamma=3.3, sigma_a=0.07, sigma_b=0.09):
    """TMA frequency spectrum (Bouws et al., 1985).

    Args:
        freq (DataArray, 1darray): Frequency array (Hz).
        hs (DataArray, float): Significant wave height (m).
        tp (DataArray, float): Peak wave period (s).
        dep (DataArray, float): Water depth (m).
        alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        gamma (DataArray, float): Peak enhancement parameter.
        sigma_a (float): width of the peak enhancement parameter for f <= fp.
        sigma_b (float): width of the peak enhancement parameter for f > fp.

    Returns:
        efth (SpecArray): TMA frequency spectrum E(f) (m2s).

    Note:
        If two or more input args other than freq are DataArrays,
            they must share the same coordinates.

    """
    check_same_coordinates(hs, tp, dep)

    dsout = jonswap(freq, hs, tp, alpha, gamma, sigma_a, sigma_b)
    k = wavenuma(freq, dep)
    phi = np.tanh(k * dep) ** 2 / (1 + (2 * k * dep) / np.sinh(2 * k * dep))
    dsout = dsout * phi

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout
