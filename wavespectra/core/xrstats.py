"""Peak based wave stats ufuncs."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core.npstats import dpm, dpm_gufunc, dp, dp_gufunc, tps, tp


def peak_wave_direction(dset):
    """Peak wave direction Dp.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.

    Returns:
        - dp (xr.DataArray): Peak wave direction data array.

    """
    # Ensure DataArray
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]

    # Dimensions checking
    if attrs.DIRNAME not in dset.dims:
        raise ValueError("Cannot calculate dp from frequency spectra.")
    if attrs.FREQNAME in dset.dims:
        dset = dset.sum(attrs.FREQNAME)

    # Peak
    ipeak = dset.argmax(dim=attrs.DIRNAME)

    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        dp_gufunc,
        ipeak,
        dset[attrs.DIRNAME],
        input_core_dims=[[], [attrs.DIRNAME]],
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "dp"
    darr.attrs = {
        "standard_name": attrs.ATTRS.dp.standard_name,
        "units": attrs.ATTRS.dp.units
    }
    return darr


def mean_direction_at_peak_wave_period(dset):
    """Mean direction at the peak wave period Dpm.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.

    Returns:
        - dpm (xr.DataArray): Mean direction at the peak wave period data array.

    Note from WW3 Manual:
        - Peak wave direction, defined like the mean direction, using the freq/wavenum
          bin containing of the spectrum F(k) that contains the peak frequency only.

    """
    # Ensure DataArray
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]

    # Dimensions checking
    if attrs.DIRNAME not in dset.dims:
        raise ValueError("Cannot calculate dp from frequency spectra.")

    # Directional moments and peaks
    msin, mcos = dset.spec.momd(1)
    ipeak = dset.spec._peak(dset.spec.oned())

    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        dpm_gufunc,
        ipeak,
        msin,
        mcos,
        input_core_dims=[[], [attrs.FREQNAME], [attrs.FREQNAME]],
        dask="parallelized",
        output_dtypes=[dset.dtype],
    )
    # Finalise
    darr.name = "dpm"
    darr.attrs = {
        "standard_name": attrs.ATTRS.dpm.standard_name,
        "units": attrs.ATTRS.dpm.units
    }
    return darr.where((darr >= 0) & (darr <= 360))


def peak_wave_period(dset, smooth=True):
    """Smooth Peak wave period Tp.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.
        - smooth (bool): Choose between the smooth (tps) or the raw (tp) peak wave period type.

    Returns:
        - tp (xr.DataArray): Peak wave period data array.

    """
    # Ensure DataArray
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]
    # Choose peak period function
    if smooth:
        func = tps
    else:
        func = tp
    # Integrate over directions
    if attrs.DIRNAME in dset.dims:
        dset = dset.sum(dim=attrs.DIRNAME)
    # Vectorize won't work if dataset does not have dims other than (freq, dir)
    if set(dset.dims) - {attrs.FREQNAME, attrs.DIRNAME}:
        vectorize = True
    else:
        vectorize = False
    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        func,
        dset,
        dset[attrs.FREQNAME],
        input_core_dims=[[attrs.FREQNAME], [attrs.FREQNAME]],
        vectorize=vectorize,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "tp"
    darr.attrs = {
        "standard_name": attrs.ATTRS.tp.standard_name,
        "units": attrs.ATTRS.tp.units
    }
    return darr


if __name__ == "__main__":

    import datetime
    from dask.diagnostics.progress import ProgressBar
    from wavespectra import read_wavespectra

    dset = read_wavespectra("/source/consultancy/jogchum/route/route_feb21/p04/spec.nc")

    ds = dset.chunk({"time": 10000})
    ds = xr.concat(50*[ds], "newdim")

    # t = datetime.datetime(1980, 4, 9, 12)
    # dsi = ds.sel(time=t)

    # tp2 = ds.spec.tp2().load()
    # tp = ds.spec.tp().load()

    # ds = ds.isel(time=0).load()
    # ds = ds.isel(time=slice(None, 100)).load()

    # print("old method")
    # with ProgressBar():
    #     dp1 = ds.spec.dpm().load()

    print("New method")
    with ProgressBar():
        dp2 = ds.spec.dp().load()

    # print("new function")
    # with ProgressBar():
    #     dp2 = mean_direction_at_peak_wave_period(ds).load()
    # print(dp2[0].values)

    # print(f"{dp1.values} vs {dp2.values}")
