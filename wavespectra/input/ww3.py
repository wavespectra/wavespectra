"""Read Native WW3 spectra files."""
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import D2R
from wavespectra.input import open_netcdf_or_zarr, to_keep


MAPPING = {
    "time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "station": attrs.SITENAME,
    "efth": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "wnddir": attrs.WDIRNAME,
    "wnd": attrs.WSPDNAME,
}


def read_ww3(filename_or_fileglob, file_format="netcdf", mapping=MAPPING, chunks={}):
    """Read Spectra from WAVEWATCHIII native netCDF format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - file_format (str): format of file to open, one of `netcdf` or `zarr`.
        - mapping (dict): coordinates mapping from original dataset to wavespectra.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation). Dimension names from original dataset or
          from wavespectra can be used to specify chunks dict.

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
    return from_ww3(dset)


def from_ww3(dset):
    """Format WW3 dataset to receive wavespectra accessor.

    Args:
        dset (xr.Dataset): Dataset created from a WW3 file.

    Returns:
        Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """
    dset = dset.rename(MAPPING)
    # Ensuring lon,lat are not function of time
    if attrs.TIMENAME in dset[attrs.LONNAME].dims:
        dset[attrs.LONNAME] = dset[attrs.LONNAME].isel(drop=True, **{attrs.TIMENAME: 0})
        dset[attrs.LATNAME] = dset[attrs.LATNAME].isel(drop=True, **{attrs.TIMENAME: 0})
    # Only selected variables to be returned
    to_drop = list(set(dset.data_vars.keys()) - to_keep)
    # Converting from radians
    dset[attrs.SPECNAME] *= D2R
    # Convert to coming-from
    dset = dset.assign_coords({attrs.DIRNAME: (dset[attrs.DIRNAME] + 180) % 360})
    # Setting standard attributes
    set_spec_attributes(dset)
    return dset.drop_vars(to_drop).drop_dims("string16", errors="ignore")
