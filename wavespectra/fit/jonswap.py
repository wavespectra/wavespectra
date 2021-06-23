"""Jonswap spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates
from wavespectra.core.attributes import attrs


def jonswap(hs, tp, freq, alpha=0.0081, gamma=3.3, sigma_a=0.07, sigma_b=0.09):
    """Jonswap frequency spectrum (Hasselmann et al., 1973).

    Args:
        hs (DataArray, float): Significant wave height (m).
        tp (DataArray, float): Peak wave period (s).
        freq (DataArray): Frequency array (Hz).
        alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        gamma (DataArray, float): Peak enhancement parameter.
        sigma_a (float): width of the peak enhancement parameter for f <= fp.
        sigma_b (float): width of the peak enhancement parameter for f > fp.

    Returns:
        efth (SpecArray): One-dimension Pierson-Moskowitz spectrum E(f) (m2s).

    Note:
        If two or more input args are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(hs, tp)

    fp = 1 / tp
    sigma = xr.full_like(freq, sigma_a).where(freq <= fp, sigma_b)
    term1 = alpha * g ** 2 * (2 * pi) ** -4 * freq ** -5
    term2 = np.exp(-(5 / 4) * (freq / fp) ** -4)
    term3 = gamma ** np.exp(-((freq - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))
    dsout = term1 * term2 * term3

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout
