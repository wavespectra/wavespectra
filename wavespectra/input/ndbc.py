"""Read NDBC netCDF spectra files.

https://dods.ndbc.noaa.gov/

"""
import warnings
import xarray as xr
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import D2R


MAPPING = {
    "time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "spectral_wave_density": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "depth": attrs.DEPNAME,
}


def read_ndbc(url, dd=10.0, always_2d=False, chunks={}):
    """Read Spectra from NDBC netCDF format.

    This function returns 2D directional spectra if directional variables are available
        and have no missing values, otherwise 1D frequency spectra are returned.

    Args:
        - url (str): Thredds URL or local path of file to read.
        - dd (float): Directional resolution for 2D spectra (deg).
        - always_2d (bool): Force it to return 2D spectra even when missing values are
          detected in any of the directional variables in the input netcdf.
        - chunks (dict): Chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions.

    Returns:
        - dset (SpecDataset): spectra dataset object read from NDBC file.

    Note:
        - Forcing to read spectra as 2D by setting `always_2d=True` will yield missing
          values in efth for times when any missing value exists in the input dataset.
        - If file is large to fit in memory, consider specifying chunks for
          'time' or other non-spectral dims.

    """
    dset = xr.open_dataset(url).chunk(chunks)
    return from_ndbc(dset, dd=dd, always_2d=always_2d)


def _construct_spectra(ef, swd1, swd2, swr1, swr2, dir):
    """Construct 2D spectra."""
    d = 0.5 + swr1 * np.cos(D2R * (dir - swd1)) + swr2 * np.cos(2 * D2R * (dir - swd2))
    return ef * d * D2R / np.pi


def from_ndbc(dset, dd=10.0, always_2d=False):
    """Format NDBC netcdf dataset to implement the wavespectra accessor.

    Args:
        - dset (xr.Dataset): Dataset created from a NDBC netcdf file.
        - dd (float): Directional resolution for 2D spectra (deg).
        - always_2d (bool): Force it to return 2D spectra even when missing values are
          detected in any of the directional variables in the input netcdf.

    Returns:
        - Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """
    has_missing = False
    if not always_2d:
        for v in ["mean_wave_dir", "principal_wave_dir", "wave_spectrum_r1", "wave_spectrum_r2"]:
            if dset[v].isnull().any():
                warnings.warn(
                    f"NaN detected in {v}, returning non-directional spectra. You can "
                    "set `always_2d=True` to ignore this and return the directional "
                    "spectra but be aware they will have missing data."
                )
                has_missing = True
                break

    dset = dset.transpose(..., "frequency")
    if has_missing:
        # Read 1D spectra if any missing values and always_2d is False
        dset = dset.spectral_wave_density
    else:
        # Construct 2D spectra
        dirs = np.arange(0, 360, dd)
        dset = _construct_spectra(
            ef=dset.spectral_wave_density,
            swd1=dset.mean_wave_dir,
            swd2=dset.principal_wave_dir,
            swr1=dset.wave_spectrum_r1,
            swr2=dset.wave_spectrum_r2,
            dir=xr.DataArray(dirs, dims=(attrs.DIRNAME,), coords={attrs.DIRNAME: dirs}),
        )
    dset = dset.to_dataset(name="efth")

    vars_and_dims = set(dset.data_vars) | set(dset.dims)
    mapping = {k: v for k, v in MAPPING.items() if k != v and k in vars_and_dims}
    dset = dset.rename(mapping)

    set_spec_attributes(dset)

    return dset
