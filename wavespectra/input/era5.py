"""Read ERA5 2D Wave Spectra NetCDF files"""

import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint

from wavespectra.core.attributes import attrs, set_spec_attributes


DEFAULT_FREQS = np.full(30, 0.03453) * (1.1 ** np.arange(0, 30))
DEFAULT_DIRS = (np.arange(7.5, 352.5 + 15, 15) + 180) % 360
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


def read_era5(filename_or_fileglob, chunks={}, freqs=None, dirs=None):
    """Read Spectra from ECMWF ERA5 netCDF format.

    Args:
        - filename_or_fileglob (str, list, filelike): filename, fileglob specifying multiple
          files, or filelike object to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - freqs (list): list of frequencies. By default use all 30 ERA5 frequencies.
        - dirs (list): list of directions. By default use all 24 ERA5 directions.

    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file.

    Note:
        - Frequency and diirection coordinates seem to have only integer positions
          which is why they are allowed to be specified as a parameter.
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """

    dset = xr.open_dataset(filename_or_fileglob).chunk(chunks)
    return from_era5(dset, freqs=freqs, dirs=dirs)


def from_era5(dset, freqs=None, dirs=None):
    """Format ERA5 netcdf dataset to receive wavespectra accessor.

    Args:
        - dset (xr.Dataset): Dataset created from a SWAN netcdf file.

    Returns:
        - Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """

    mapping = {k: v for k, v in MAPPING.items() if k != v and k in dset.variables}
    dset = dset.rename(mapping).reset_coords(drop=True)

    # Convert ERA5 format to wavespectra format
    dset[attrs.SPECNAME] = 10**dset[attrs.SPECNAME] * np.pi / 180
    dset[attrs.SPECNAME] = dset[attrs.SPECNAME].fillna(0)

    dset[attrs.FREQNAME] = freqs if freqs else DEFAULT_FREQS
    dset[attrs.DIRNAME] = dirs if dirs else DEFAULT_DIRS

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
        freqs=None,
        dirs=None,
    ):
        return read_era5(filename_or_obj, freqs=freqs, dirs=dirs)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open ERA5 netcdf spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
