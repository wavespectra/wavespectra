"""Read Native WWM netCDF spectra files."""
import xarray as xr
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import uv_to_spddir
from wavespectra.input import open_netcdf_or_zarr, to_keep

R2D = 180 / np.pi

MAPPING = {
    "nfreq": attrs.FREQNAME,
    "ndir": attrs.DIRNAME,
    "nbstation": attrs.SITENAME,
    "AC": attrs.SPECNAME,
    "lon": attrs.LONNAME,
    "lat": attrs.LATNAME,
    "DEP": attrs.DEPNAME,
    "ocean_time": attrs.TIMENAME,
}


def read_wwm(filename_or_fileglob, file_format="netcdf", mapping=MAPPING, chunks={}):
    """Read Spectra from WWMII native netCDF format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - file_format (str): format of file to open, one of `netcdf` or `zarr`.
        - mapping (dict): coordinates mapping from original dataset to wavespectra.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).

    Returns:
        - dset (SpecDataset): spectra dataset object read from ww3 file.

    Note:
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    dset = open_netcdf_or_zarr(
        filename_or_fileglob=filename_or_fileglob,
        file_format=file_format,
        mapping=mapping,
        chunks=chunks,
    )
    return from_wwm(dset)


def from_wwm(dset):
    """Format WWM netcdf dataset to receive wavespectra accessor.

    Args:
        dset (xr.Dataset): Dataset created from a SWAN netcdf file.

    Returns:
        Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """
    dset = dset.rename(MAPPING)
    # Calculating wind speeds and directions
    if "Uwind" in dset and "Vwind" in dset:
        dset[attrs.WSPDNAME], dset[attrs.WDIRNAME] = uv_to_spddir(
            dset["Uwind"], dset["Vwind"], coming_from=True
        )
    # Assigning spectral coordinates
    dset = dset.assign_coords({attrs.FREQNAME: dset.SPSIG / (2 * np.pi)})
    dset = dset.assign_coords({attrs.DIRNAME: dset.SPDIR * R2D})
    # Setting standard attributes
    set_spec_attributes(dset)
    # converting Action to Energy density and adjust density to Hz
    dset[attrs.SPECNAME] = dset[attrs.SPECNAME] * dset.SPSIG * (2 * np.pi) / R2D
    # Returns only selected variables, transposed
    to_drop = list(set(dset.data_vars.keys()) - to_keep)
    dims = [d for d in ["time", "site", "freq", "dir"] if d in dset.efth.dims]
    return dset.drop_vars(to_drop).transpose(*dims)
