"""Jonswap spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra.core.utils import scaled, check_same_coordinates, to_coords
from wavespectra.core.attributes import attrs
from wavespectra.core.npstats import hs_numpy


def fit_jonswap(
    freq, fp, alpha=0.0081, gamma=3.3, sigma_a=0.07, sigma_b=0.09, hs=None, **kwargs
):
    """Jonswap frequency spectrum for developing seas (Hasselmann et al., 1973).

    Args:
        - freq (DataArray, 1darray, list): Frequency array (Hz).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        - gamma (DataArray, float): Peak enhancement parameter.
        - sigma_a (DataArray, float): width of the peak enhancement parameter for f <= fp.
        - sigma_b (DataArray, float): width of the peak enhancement parameter for f > fp.
        - hs (DataArray, float): Significant wave height (m), if provided the Jonswap
          spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.

    Returns:
        - efth (SpecArray): Jonswap spectrum E(f) (m2s).

    Note:
        - If `hs` is provided than the scaling parameter `alpha` becomes irrelevant.
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(fp, alpha, gamma, sigma_a, sigma_b, hs)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    sigma = xr.where(freq <= fp, sigma_a, sigma_b)
    term1 = alpha * g ** 2 * (2 * pi) ** -4 * freq ** -5
    term2 = np.exp(-(5 / 4) * (freq / fp) ** -4)
    term3 = gamma ** np.exp(-((freq - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))
    dsout = term1 * term2 * term3

    if hs is not None:
        dsout = scaled(dsout, hs)

    dsout.name = attrs.SPECNAME

    return dsout


def np_jonswap(
    freq, fp, alpha=0.0081, gamma=3.3, sigma_a=0.07, sigma_b=0.09, hs=None
):
    """Jonswap frequency spectrum for developing seas (Hasselmann et al., 1973).

    Args:
        - freq (1darray): Frequency array (Hz).
        - fp (float): Peak wave frequency (Hz).
        - alpha (float): Phillip's fetch-dependent scaling coefficient.
        - gamma (float): Peak enhancement parameter.
        - sigma_a (float): width of the peak enhancement parameter for f <= fp.
        - sigma_b (float): width of the peak enhancement parameter for f > fp.
        - hs (float): Significant wave height (m), if provided the Jonswap
          spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.

    Returns:
        - efth (SpecArray): Jonswap spectrum E(f) (m2s).

    Note:
        - If `hs` is provided than the scaling parameter `alpha` becomes irrelevant.
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    sigma = np.where(freq <= fp, sigma_a, sigma_b)
    term1 = alpha * g ** 2 * (2 * pi) ** -4 * freq ** -5
    term2 = np.exp(-(5 / 4) * (freq / fp) ** -4)
    term3 = gamma ** np.exp(-((freq - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))
    dsout = term1 * term2 * term3

    if hs is not None:
        hs_jonswap = hs_numpy(dsout, freq)
        dsout = dsout * (hs / hs_jonswap)**2

    return dsout