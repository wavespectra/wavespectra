"""Peak based wave stats ufuncs."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core import npstats


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

    # Ensure single chunk along input core dimensions
    dset = dset.chunk({attrs.DIRNAME: None})

    # Peak
    ipeak = dset.argmax(dim=attrs.DIRNAME)

    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        npstats.dp_gufunc,
        ipeak.astype("int64"),
        dset[attrs.DIRNAME].astype("float32"),
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

    # Ensure single chunk along input core dimensions
    dset = dset.chunk({attrs.FREQNAME: None})

    # Directional moments and peaks
    msin, mcos = dset.spec.momd(1)
    ipeak = dset.spec._peak(dset.spec.oned())

    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        npstats.dpm_gufunc,
        ipeak.astype("int64"),
        msin.astype("float64"),
        mcos.astype("float64"),
        input_core_dims=[[], [attrs.FREQNAME], [attrs.FREQNAME]],
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "dpm"
    darr.attrs = {
        "standard_name": attrs.ATTRS.dpm.standard_name,
        "units": attrs.ATTRS.dpm.units
    }
    return darr #.where((darr >= 0) & (darr <= 360))


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
        func = npstats.tps_gufunc
    else:
        func = npstats.tp_gufunc

    # Integrate over directions
    if attrs.DIRNAME in dset.dims:
        dset = dset.sum(dim=attrs.DIRNAME)

    # Ensure single chunk along input core dimensions
    dset = dset.chunk({attrs.FREQNAME: None})

    # Frequency Peaks
    ipeak = dset.spec._peak(dset)

    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        func,
        ipeak.astype("int64"),
        dset.astype("float64"),
        dset[attrs.FREQNAME].astype("float32"),
        input_core_dims=[[], [attrs.FREQNAME], [attrs.FREQNAME]],
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
