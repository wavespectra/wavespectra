"""Gaussian spectrum."""
import numpy as np
import xarray as xr
from scipy.constants import pi

from wavespectra import SpecArray
from wavespectra.core.utils import scaled, check_same_coordinates, to_coords, load_function
from wavespectra.core.attributes import attrs


def fit_gaussian_conditional(freq, hs, fp, gw, tm01, tm02, rt_thres=0.95, otherwise='fit_tma', **kwargs):
    """Conditional Gaussian frequency spectrum, only used if 1D symmetric in frequency (Bunney et al., 2014).

    Args:
        - freq (DataArray): Frequency array (Hz).
        - hs (DataArray, float): Significant wave height (m).
        - fp (DataArray, float): Peak wave frequency (Hz).
        - gw (DataArray, float): Gaussian width parameter :math:`\sigma`
        - tm01 (DataArray, float): First-moment mean wave period (s).
        - tm02 (DataArray, float): Second-moment mean wave period (s).
        - rt_thres (optional, float): Threshold for symmetry of frequency distribution. Default = 0.95.
        - otherwise (string): The fit function to use for rt < rt_thres. Default = 'fit_tma'


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

    t0 = 1/np.min(freq)
    rt = (tm01 - t0)/(tm02-t0)

    inds = rt > rt_thres

    # Gaussian fit
    mo = (hs / 4) ** 2
    term1 = mo / (gw * np.sqrt(2 * pi))
    term2 = np.exp(-((2 * pi * freq - 2 * pi * fp) ** 2 / (2 * (gw) ** 2)))
    ds_gaussian = term1 * term2

    ds_gaussian = scaled(ds_gaussian, hs)
    ds_gaussian.name = attrs.SPECNAME

    # Otherwise fit. Inspect the args and kwargs passed to this function
    import inspect
    arg_vals = inspect.getargvalues(inspect.currentframe())
    arguments = {a:arg_vals.locals[a] for a in arg_vals.args}
    arguments.update(arg_vals.locals['kwargs'])
    fit_func = load_function("wavespectra", otherwise, prefix="fit_")
    ds_otherwise = fit_func(**arguments)

    dsout = xr.where(inds,ds_gaussian,ds_otherwise)
    return dsout
