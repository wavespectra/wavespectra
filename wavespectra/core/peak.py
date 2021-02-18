"""Peak based wave stats ufuncs."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


def _peak(arr):
    """Index of largest peak along freq dim in a E(f,d).

    Args:
        arr (1darray): Frequency spectrum.

    Returns:
        ipeak (SpecArray): indices for slicing arr at the frequency peak

    Note:
        A peak is found when arr(ipeak-1) < arr(ipeak) < arr(ipeak+1)
        ipeak==0 does not satisfy above condition and is assumed to be
            missing_value in other parts of the code
    """
    ispeak = (np.diff(np.append(arr[0], arr)) > 0) & (
        np.diff(np.append(arr, arr[-1])) < 0
    )
    peak_pos = np.where(ispeak, arr, 0).argmax()
    return peak_pos


def np_tp(spectrum, freq):
    """Peak wave period Tp on numpy array.

    Args:
        - spectrum (2darray): Wave spectrum array E(f,d).
        - freq (1darray): Wave frequency array.

    """
    fspec = spectrum.sum(axis=1)
    ipeak = _peak(fspec)
    if not ipeak:
        return None
    import ipdb; ipdb.set_trace()
    sig1 = freq[ipeak - 1]
    sig2 = freq[ipeak + 1]
    sig3 = freq[ipeak]
    e1 = fspec[ipeak - 1]
    e2 = fspec[ipeak + 1]
    e3 = fspec[ipeak]
    p = sig1 + sig2
    q = (e1 - e2) / (sig1 - sig2)
    r = sig1 + sig3
    t = (e1 - e3) / (sig1 - sig3)
    a = (t - q) / (r - p)
    if a < 0:
        sigp = (-q + p * a) / (2.0 * a)
    else:
        sigp = sig3
    return 1.0 / sigp


def peak_wave_period(efth):
    """Peak wave period Tp.

    Args:
        - efth (xr.DataArray): Spectra array in wavespectra convention.

    Returns:
        - dspart (xr.DataArray): Partitioned spectra with extra `part` dimension.

    """
    darr = xr.apply_ufunc(
        np_tp,
        efth,
        efth.freq,
        input_core_dims=[["freq", "dir"], ["freq"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    darr.name = "tp"
    darr.attrs = {
        "standard_name": attrs.ATTRS.tp.standard_name,
        "units": attrs.ATTRS.tp.units
    }
    return darr


if __name__ == "__main__":

    from wavespectra import read_wavespectra

    dset = read_wavespectra("/source/consultancy/jogchum/route/route_feb21/p04/spec.nc")

    ds = dset.isel(time=slice(None, 1000)).chunk()

    # # Numpy
    # spectrum = ds.efth.isel(time=0).values
    # freq = ds.freq.values
    # tp = np_tp(spectrum, freq)

    # # Existing method
    # tp_old = ds.spec.tp()

    # Xarray ufunc
    ds = dset.isel(time=0).load()
    ds.attrs = {}
    tp_new = peak_wave_period(ds.efth)