"""Read ERA5 2D Wave Spectra NetCDF files"""

import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint

from wavespectra.core.utils import create_frequencies
from wavespectra.core.attributes import attrs, set_spec_attributes


MAPPING = {
    "time": attrs.TIMENAME,
    "valid_time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "frequencyNumber": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "directionNumber": attrs.DIRNAME,
    "d2fd": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
}


def read_era5(filename_or_fileglob, chunks={}, f0=0.03453, df=1.1):
    """Read Spectra from ECMWF ERA5 netCDF format.

    Args:
        - filename_or_fileglob (str, list, filelike): filename, fileglob specifying multiple
          files, or filelike object to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - f0 (float): First frequency value in Hz (e.g., 0.03453)
        - df (float): Multiplicative increment between frequencies.

    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file.

    Note:
        - This reader also supports ECMWF spectra from the operational forecast.
        - Documentation describing how to construct the spectral grid can be found at
          https://www.ecmwf.int/en/forecasts/documentation-and-support/2d-wave-spectra.
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'latitude/longitude' dims.

    """

    dset = xr.open_dataset(filename_or_fileglob).chunk(chunks)
    return from_era5(dset, f0=f0, df=df)


def from_era5(dset, f0=0.03453, df=1.1):
    """Format ERA5 netcdf dataset to receive wavespectra accessor.

    Args:
        - dset (xr.Dataset): Dataset created from a ERA5 netcdf file.
        - f0 (float): First frequency value in Hz (e.g., 0.03453)
        - df (float): Multiplicative increment between frequencies.

    Returns:
        - Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """

    mapping = {k: v for k, v in MAPPING.items() if k != v and k in dset.variables}
    dset = dset.rename(mapping).reset_coords(drop=True)

    # Convert ERA5 format to wavespectra format
    dset[attrs.SPECNAME] = 10**dset[attrs.SPECNAME] * np.pi / 180
    dset[attrs.SPECNAME] = dset[attrs.SPECNAME].fillna(0)

    # Assign the logarithmic frequency grid
    dset[attrs.FREQNAME] = create_frequencies(f0, dset[attrs.FREQNAME].size, df)

    # Assign the coming-from direction grid
    dd = 360 / dset[attrs.DIRNAME].size
    dset[attrs.DIRNAME] = ((np.arange(0, 360, dd) + dd / 2) + 180) % 360

    # Setting standard attributes
    set_spec_attributes(dset)

    return dset


class ERA5BackendEntrypoint(BackendEntrypoint):
    """ERA5 backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        f0=0.03453,
        df=1.1,
    ):
        return read_era5(filename_or_obj, f0=f0, df=df)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open ERA5 netcdf spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
