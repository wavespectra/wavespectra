"""Read customised MetOcean Solutions WW3 spectra files."""
import xarray as xr
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes


def read_ww3_msl(filename_or_fileglob, chunks={}, file_format="netcdf"):
    """Read Spectra from WAVEWATCHIII MetOcean Solutions netCDF format.

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
          'time' and/or 'site' dims

    """
    if file_format == "netcdf":
        dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks, combine="by_coords")
    elif file_format == "zarr":
        fsmap = get_mapper(filename_or_fileglob)
        dset = xr.open_zarr(fsmap, consolidated=True, chunks=chunks)
    else:
        raise ValueError("file_format must be one of ('netcdf', 'zarr')")
    return from_ww3_msl(dset)


def from_ww3_msl(dset):
    """Format WW3-MSL netcdf dataset to receive wavespectra accessor.

    Args:
        dset (xr.Dataset): Dataset created from a SWAN netcdf file.

    Returns:
        Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """
    _units = dset.specden.attrs.get("units", "")
    dset = dset.rename(
        {"freq": attrs.FREQNAME, "dir": attrs.DIRNAME, "wsp": attrs.WSPDNAME}
    )
    dset[attrs.SPECNAME] = (dset["specden"].astype("float32") + 127.0) * dset["factor"]
    dset = dset.drop(["specden", "factor", "df"])
    # Assign site coordinate so they will look like those read from native ww3 files
    dset[attrs.SITENAME] = np.arange(1.0, dset.site.size + 1)
    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update({"_units": _units, "_variable_name": "specden"})
    if attrs.DIRNAME not in dset or len(dset.dir) == 1:
        dset[attrs.SPECNAME].attrs.update({"units": "m^{2}.s"})
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
    return dset.drop(to_drop)