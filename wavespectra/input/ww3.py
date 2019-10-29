"""Read Native WW3 spectra files."""
import xarray as xr
import numpy as np
from fsspec import get_mapper

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes

D2R = np.pi / 180


def read_ww3(filename_or_fileglob, chunks={}, file_format="netcdf"):
    """Read Spectra from WAVEWATCHIII native netCDF format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - file_format (str): format of file to open, one of `netcdf` or `zarr`.

    Returns:
        - dset (SpecDataset): spectra dataset object read from ww3 file.

    Note:
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    if file_format == "netcdf":
        dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks, combine="by_coords")
    elif file_format == "zarr":
        fsmap = get_mapper(filename_or_fileglob)
        dset = xr.open_zarr(fsmap, consolidated=True, chunks=chunks)
    else:
        raise ValueError("file_format must be one of ('netcdf', 'zarr')")
    return from_ww3(dset)


def from_ww3(dset):
    """Format WW3 dataset to receive wavespectra accessor.

    Args:
        dset (xr.Dataset): Dataset created from a WW3 file.

    Returns:
        Formated dataset with the SpecDataset accessor in the `spec` namespace.

"""
    _units = dset.efth.attrs.get("units", "")
    dset = dset.rename(
        {
            "frequency": attrs.FREQNAME,
            "direction": attrs.DIRNAME,
            "station": attrs.SITENAME,
            "efth": attrs.SPECNAME,
            "longitude": attrs.LONNAME,
            "latitude": attrs.LATNAME,
            "wnddir": attrs.WDIRNAME,
            "wnd": attrs.WSPDNAME,
        }
    )
    # Ensuring lon,lat are not function of time
    if attrs.TIMENAME in dset[attrs.LONNAME].dims:
        dset[attrs.LONNAME] = dset[attrs.LONNAME].isel(drop=True, **{attrs.TIMENAME: 0})
        dset[attrs.LATNAME] = dset[attrs.LATNAME].isel(drop=True, **{attrs.TIMENAME: 0})
    # Only selected variables to be returned
    to_drop = [
        dvar
        for dvar in dset.data_vars
        if dvar
        not in [
            attrs.SPECNAME,
            attrs.WSPDNAME,
            attrs.WDIRNAME,
            attrs.DEPNAME,
            attrs.LONNAME,
            attrs.LATNAME,
        ]
    ]
    # Converting from radians
    dset[attrs.SPECNAME] *= D2R
    # Setting standard names and storing original file attributes
    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update(
        {"_units": _units, "_variable_name": attrs.SPECNAME}
    )
    # Adjustting attributes if 1D
    if attrs.DIRNAME not in dset or len(dset.dir) == 1:
        dset[attrs.SPECNAME].attrs.update({"units": "m^{2}.s"})
    dset[attrs.DIRNAME] = (dset[attrs.DIRNAME] + 180) % 360
    return dset.drop(to_drop)
