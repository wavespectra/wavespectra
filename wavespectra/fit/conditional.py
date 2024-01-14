"""Conditional frequency spectrum selecting a given shape based on a boolean array."""
import xarray as xr

from wavespectra.core.utils import (
    check_same_coordinates,
    to_coords,
    load_function,
)
from wavespectra.core.attributes import attrs


def fit_conditional(
    freq, hs, fp, cond, when_true="fit_jonswap", when_false="fit_gaussian", **kwargs
):
    """Conditional frequency spectrum.

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

    true_func = load_function("wavespectra", when_true, prefix="fit_")
    false_func = load_function("wavespectra", when_false, prefix="fit_")

    ds_true = true_func(**arguments)
    ds_false = false_func(**arguments)
    dsout = xr.where(cond, ds_true, ds_false)

    dsout.name = attrs.SPECNAME

    return dsout
