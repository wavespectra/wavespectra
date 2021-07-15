"""Pierson and Moskowitz spectrum."""
import numpy as np
import xarray as xr

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates, to_coords
from wavespectra.core.attributes import attrs


def fit_pierson_moskowitz(freq, hs, tp, scale_by_hs=True, **kwargs):
    """Pierson and Moskowitz spectrum for fully developed seas (Pierson and Moskowitz, 1964).

    Args:
        - freq (DataArray, 1darray, list): Frequency array (Hz).
        - hs (DataArray, float): Significant wave height (m).
        - tp (DataArray, float): Peak wave period (s).
        - scale_by_hs (bool): Scale spectra so the variance corresponds to hs.

    Returns:
        - efth (SpecArray): Pierson-Moskowitz frequency spectrum E(f) (m2s).

    Note:
        - The spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.
        - If `hs` and `tp` args are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(hs, tp)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    b = (tp / 1.057) ** -4
    a = b * (hs / 2) ** 2
    dsout = a * freq ** -5 * np.exp(-b * freq ** -4)

    if scale_by_hs:
        dsout = scaled(dsout, hs)

    dsout.name = attrs.SPECNAME

    return dsout
