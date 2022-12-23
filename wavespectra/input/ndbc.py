"""Read NDBC netCDF spectra files.

https://dods.ndbc.noaa.gov/

"""
import xarray as xr
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import uv_to_spddir, R2D
from wavespectra.input import open_netcdf_or_zarr, to_keep


MAPPING = {
    "time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "spectral_wave_density": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "depth": attrs.DEPNAME,
}


def read_ndbc(url, chunks={}):
    """Read Spectra from NDBC netCDF format.

    Args:
        - url (str): Thredds URL or local path of file to read.
        - mapping (dict): Coordinates mapping from original dataset to wavespectra.
        - chunks (dict): Chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions.

    Returns:
        - dset (SpecDataset): spectra dataset object read from NDBC file.

    Note:
        - If file is large to fit in memory, consider specifying chunks for
          'time' or other non-spectral dims.

    """
    dset = xr.open_dataset(url).chunk(chunks)
    return from_ndbc(dset)


def from_ndbc(dset):
    """Format NDBC netcdf dataset to implement the wavespectra accessor.

    Args:
        dset (xr.Dataset): Dataset created from a NDBC netcdf file.

    Returns:
        Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """
    vars_and_dims = set(dset.data_vars) | set(dset.dims)
    mapping = {k: v for k, v in MAPPING.items() if k != v and k in vars_and_dims}
    dset = dset.rename(mapping).transpose(..., "freq")
    # Setting standard attributes
    set_spec_attributes(dset)
    if attrs.DIRNAME not in dset[attrs.SPECNAME].dims:
        dset[attrs.SPECNAME].attrs = dict(
            standard_name="sea_surface_wave_variance_spectral_density",
            units="m2 s",
        )
    # return dset.drop_vars(to_drop)
    return dset
