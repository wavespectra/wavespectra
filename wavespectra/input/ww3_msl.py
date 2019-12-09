"""Read customised MetOcean Solutions WW3 spectra files."""
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.input import open_netcdf_or_zarr, to_keep

MAPPING = {
    "time": attrs.TIMENAME,
    "freq": attrs.FREQNAME,
    "dir": attrs.DIRNAME,
    "site": attrs.SITENAME,
    "specden": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "depth": attrs.DEPNAME,
}


def read_ww3_msl(
    filename_or_fileglob, file_format="netcdf", mapping=MAPPING, chunks={}
):
    """Read Spectra from WAVEWATCHIII MetOcean Solutions netCDF format.

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
          'time' and/or 'site' dims

    """
    dset = open_netcdf_or_zarr(
        filename_or_fileglob=filename_or_fileglob,
        file_format=file_format,
        mapping=mapping,
        chunks=chunks,
    )
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
    dset = dset.drop_vars(["specden", "factor", "df"])
    # Assign site coordinate so they will look like those read from native ww3 files
    dset[attrs.SITENAME] = np.arange(1.0, dset.site.size + 1)
    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update({"_units": _units, "_variable_name": "specden"})
    if attrs.DIRNAME not in dset or len(dset.dir) == 1:
        dset[attrs.SPECNAME].attrs.update({"units": "m^{2}.s"})
    # Only selected variables to be returned
    to_drop = list(set(dset.data_vars.keys()) - to_keep)
    return dset.drop_vars(to_drop)
