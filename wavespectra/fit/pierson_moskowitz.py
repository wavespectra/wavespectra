"""Pierson and Moskowitz spectrum."""
import numpy as np

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates
from wavespectra.core.attributes import attrs


def pierson_moskowitz(freq, hs, tp):
    """Pierson and Moskowitz (Pierson and Moskowitz, 1964).

    Args:
        freq (DataArray): Frequency array (Hz).
        hs (DataArray, float): Significant wave height (m).
        tp (DataArray, float): Peak wave period (s).

    Returns:
        efth (SpecArray): Pierson-Moskowitz frequency spectrum E(f) (m2s).

    Note:
        If `hs` and `tp` args are DataArrays they must share the same coordinates.

    """
    check_same_coordinates(hs, tp)

    b = (tp / 1.057) ** -4
    a = b * (hs / 2) ** 2
    dsout = a * freq ** -5 * np.exp(-b * freq ** -4)

    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME

    return dsout
