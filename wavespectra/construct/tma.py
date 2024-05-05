"""TMA spectrum."""
import numpy as np
import xarray as xr

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates, wavenuma, to_coords
from wavespectra.core.attributes import attrs
from wavespectra.fit.jonswap import fit_jonswap


def fit_tma(
    freq,
    fp,
    dep,
    alpha=0.0081,
    gamma=3.3,
    sigma_a=0.07,
    sigma_b=0.09,
    hs=None,
    **kwargs,
):
    """TMA frequency spectrum for seas in water of finite depth (Bouws et al., 1985).

    Args:
        - freq (DataArray, 1darray, list): Frequency array (Hz).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - dep (DataArray, float): Water depth (m).
        - alpha (DataArray, float): Phillip's fetch-dependent scaling coefficient.
        - gamma (DataArray, float): Peak enhancement parameter.
        - sigma_a (float): width of the peak enhancement parameter for f <= fp.
        - sigma_b (float): width of the peak enhancement parameter for f > fp.
        - hs (DataArray, float): Significant wave height (m), if provided the Jonswap
          spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.

    Returns:
        - efth (SpecArray): TMA frequency spectrum E(f) (m2s).

    Note:
        - If `hs` is provided than the scaling parameter `alpha` becomes irrelevant.
        - If two or more input args other than freq are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(fp, dep, alpha, gamma, sigma_a, sigma_b, hs)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    dsout = fit_jonswap(freq, fp, alpha, gamma, sigma_a, sigma_b, hs)
    k = wavenuma(freq, dep)
    phi = np.tanh(k * dep) ** 2 / (1 + (2 * k * dep) / np.sinh(2 * k * dep))
    dsout = dsout * phi

    if hs is not None:
        dsout = scaled(dsout, hs)

    dsout.name = attrs.SPECNAME

    return dsout
