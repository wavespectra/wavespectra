"""Peak based wave stats ufuncs."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core.npstats import tps, tp


def peak_wave_period(dset, smooth=True):
    """Smooth Peak wave period Tp.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.
        - smooth (bool): Choose between the smooth (tps) or the raw (tp) peak wave period type.

    Returns:
        - tp (xr.DataArray): Peak wave period data array.

    """
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]
    if smooth:
        func = tps
    else:
        func = tp
    darr = xr.apply_ufunc(
        func,
        dset.sum(dim=attrs.DIRNAME),
        dset[attrs.FREQNAME],
        input_core_dims=[[attrs.FREQNAME], [attrs.FREQNAME]],
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

    ds = dset.chunk({"time": 10000})

    # # Existing method
    # tp_old = ds.spec.tp()

    # Xarray ufunc
    tp_new = peak_wave_period(ds.efth)