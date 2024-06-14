"""Frequency spectral shapes."""
import numpy as np
import xarray as xr
from scipy.constants import g, pi

from wavespectra.core.utils import scaled, check_same_coordinates, wavenuma, to_coords
from wavespectra.core.attributes import attrs
from wavespectra.core import npstats


def pierson_moskowitz(freq, fp, alpha=0.0081, hs=None, **kwargs):
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
        - If `hs` is provided the spectra are scaled so that :math:`4\\sqrt{m_0} = hs`
          and the scaling parameter `alpha` becomes irrelevant.
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(fp, alpha)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    dsout = (
        alpha * g**2 / (2 * pi) ** 4 / freq**5 * np.exp(-1.25 * (freq / fp) ** -4)
    )

    if hs is not None:
        dsout = scaled(dsout, hs)

    dsout.name = attrs.SPECNAME

    return dsout


def jonswap(
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
    term1 = alpha * g**2 * (2 * pi) ** -4 * freq**-5
    term2 = np.exp(-(5 / 4) * (freq / fp) ** -4)
    term3 = gamma ** np.exp(-((freq - fp) ** 2) / (2 * sigma**2 * fp**2))
    dsout = term1 * term2 * term3

    if hs is not None:
        dsout = scaled(dsout, hs)

    dsout.name = attrs.SPECNAME

    return dsout


def tma(
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

    dsout = jonswap(freq, fp, alpha, gamma, sigma_a, sigma_b, hs)
    k = wavenuma(freq, dep)
    phi = np.tanh(k * dep) ** 2 / (1 + (2 * k * dep) / np.sinh(2 * k * dep))
    dsout = dsout * phi

    if hs is not None:
        dsout = scaled(dsout, hs)

    dsout.name = attrs.SPECNAME

    return dsout


def gaussian(freq, hs, fp, gw, **kwargs):
    """Gaussian frequency spectrum (Bunney et al., 2014).

    Args:
        - freq (DataArray): Frequency array (Hz).
        - hs (DataArray, float): Significant wave height (m).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - gw (DataArray, float): Gaussian width parameter :math:`\sigma` (m2s).

    Returns:
        - efth (SpecArray): Gaussian frequency spectrum E(f) (m2s).

    Note:
        - The spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(hs, fp, gw)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    mo = (hs / 4) ** 2
    dsout = mo / (gw * np.sqrt(2 * pi)) * np.exp(-0.5 * ((freq - fp) / gw) ** 2)
    dsout = scaled(dsout, hs)
    dsout.name = attrs.SPECNAME
    return dsout


def conditional(
    freq, hs, fp, cond, when_true="jonswap", when_false="gaussian", **kwargs
):
    """Conditional frequency spectrum selecting a given shape based on a boolean array.

    Args:
        - freq (DataArray): Frequency array (Hz).
        - hs (DataArray, float): Significant wave height (m).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - cond (DataArray, bool): .
        - when_true: spectral shape function when cond == True.
        - when_false: spectral shape function when cond == False.
        - kwargs: kwargs must have all arguments for both when_true and when_false functions.

    Returns:
        - efth (SpecArray): Conditional frequency spectrum E(f) (m2s).

    Note:
        - The spectra are scaled so that :math:`4\\sqrt{m_0} = hs`.
        - If two or more input args other than `freq` are DataArrays,
          they must share the same coordinates.

    """
    check_same_coordinates(hs, fp, cond)
    if not isinstance(freq, xr.DataArray):
        freq = to_coords(freq, "freq")

    import inspect

    arg_vals = inspect.getargvalues(inspect.currentframe())
    arguments = {a: arg_vals.locals[a] for a in arg_vals.args}
    arguments.update(arg_vals.locals["kwargs"])

    true_func = globals()[when_true]
    false_func = globals()[when_false]

    ds_true = true_func(**arguments)
    ds_false = false_func(**arguments)
    dsout = xr.where(cond, ds_true, ds_false)

    dsout.name = attrs.SPECNAME

    return dsout
