"""Peak based wave stats ufuncs."""
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
        npstats.dp,
        ipeak.astype("int64"),
        dset[attrs.DIRNAME].astype("float32"),
        input_core_dims=[[], [attrs.DIRNAME]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "dp"
    darr.attrs = {
        "standard_name": attrs.ATTRS.dp.standard_name,
        "units": attrs.ATTRS.dp.units,
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
        npstats.dpm,
        ipeak.astype("int64"),
        msin.astype("float64"),
        mcos.astype("float64"),
        input_core_dims=[[], [attrs.FREQNAME], [attrs.FREQNAME]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "dpm"
    darr.attrs = {
        "standard_name": attrs.ATTRS.dpm.standard_name,
        "units": attrs.ATTRS.dpm.units,
    }
    return darr  # .where((darr >= 0) & (darr <= 360))


def alpha(dset, smooth=True):
    """Jonswap fetch dependant scaling coefficient alpha.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.
        - smooth (bool): Choose between the smooth or discrete peak frequency.

    Returns:
        - alpha (xr.DataArray): Peak wave period data array.

    """
    # Ensure DataArray
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]

    # Peak frequency
    fp = 1 / peak_wave_period(dset, smooth=smooth)

    # Ensure single chunk along input core dimensions
    dset = dset.chunk({attrs.FREQNAME: None})

    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        npstats.alpha,
        dset.astype("float64"),
        dset[attrs.FREQNAME].astype("float32"),
        fp.astype("float32"),
        input_core_dims=[[attrs.FREQNAME], [attrs.FREQNAME], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "alpha"
    darr.attrs = {
        "standard_name": attrs.ATTRS.alpha.standard_name,
        "units": attrs.ATTRS.alpha.units,
    }
    return darr


def peak_wave_period(dset, smooth=True):
    """Smooth Peak wave period Tp.

    Args:
        - dset (xr.DataArray, xr.Dataset): 1D spectra array or dataset in wavespectra convention.
        - smooth (bool): Choose between the smooth (tps) or the raw (tp) peak wave period type.

    Returns:
        - tp (xr.DataArray): Peak wave period data array.

    """
    # Ensure DataArray
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]

    # Choose peak period function
    if smooth:
        func = npstats.tps
    else:
        func = npstats.tp

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
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "tp"
    darr.attrs = {
        "standard_name": attrs.ATTRS.tp.standard_name,
        "units": attrs.ATTRS.tp.units,
    }
    return darr


def peak_directional_spread(dset, mom=1):
    """Wave spreading at the peak wave frequency Dpspr.

    Args:
        - dset (xr.DataArray, xr.Dataset): Spectra array or dataset in wavespectra convention.
        - mom (int): Directional moment for calculating the mth directional spread.

    Returns:
        - dpspr (xr.DataArray): Wave spreading at the peak wave frequency data array.

    """
    # Ensure DataArray
    if isinstance(dset, xr.Dataset):
        dset = dset[attrs.SPECNAME]

    # Dimensions checking
    if attrs.DIRNAME not in dset.dims:
        raise ValueError("Cannot calculate dpspr from frequency spectra.")

    # Ensure single chunk along input core dimensions
    dset = dset.chunk({attrs.FREQNAME: None})

    # Frequency dependant directional spread and frequency peaks
    fdspr = dset.spec.fdspr(mom=mom)
    ipeak = dset.spec._peak(dset.spec.oned())

    # Apply function over the full dataset
    darr = xr.apply_ufunc(
        npstats.dpspr,
        ipeak.astype("int64"),
        fdspr.astype("float64"),
        input_core_dims=[[], [attrs.FREQNAME]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    # Finalise
    darr.name = "dpspr"
    darr.attrs = {
        "standard_name": attrs.ATTRS.dpspr.standard_name,
        "units": attrs.ATTRS.dpspr.units,
    }
    return darr
