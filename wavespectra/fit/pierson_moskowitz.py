"""Pierson and Moskowitz spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates, to_coords
from wavespectra.core.attributes import attrs


def fit_pierson_moskowitz(freq, fp, alpha=0.0081, hs=None, **kwargs):
    """Pierson and Moskowitz spectrum for fully developed seas (Pierson and Moskowitz, 1964).

    Args:
        - freq (DataArray, 1darray, list): Frequency array (Hz).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        - hs (DataArray, float): Significant wave height (m), if provided the spectra
          are scaled so that :math:`4\\sqrt{m_0} = hs`.

    Returns:
        - efth (SpecArray): Pierson-Moskowitz frequency spectrum E(f) (m2s).

    Note:
        - If `hs` is provided than the scaling parameter `alpha` becomes irrelevant.
        - The spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(fp, alpha)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    dsout = alpha * g**2 / (2 * pi)**4 / freq**5 * np.exp(-1.25 * (freq / fp) ** -4)

    if hs is not None:
        dsout = scaled(dsout, hs)

    dsout.name = attrs.SPECNAME

    return dsout
