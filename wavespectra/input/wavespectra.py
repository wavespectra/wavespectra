"""Read netCDF or ZARR formatted with wavespectra conventions."""
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.input import open_netcdf_or_zarr


def read_wavespectra(
    filename_or_fileglob,
    file_format="netcdf",
    chunks={},
):
    """Read Spectra from from netCDF or ZARR format in Wavespectra convention.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - file_format (str): format of file to open, one of `netcdf` or `zarr`.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).

    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file

    Note:
        - Assumes frequency in :math:`Hz`, direction in :math:`degree` and
          spectral energy in :math:`m^{2}degree^{-1}{s}`.
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    mapping = {
        timename: attrs.TIMENAME,
        freqname: attrs.FREQNAME,
        dirname: attrs.DIRNAME,
        sitename: attrs.SITENAME,
        specname: attrs.SPECNAME,
        lonname: attrs.LONNAME,
        latname: attrs.LATNAME,
        depname: attrs.DEPNAME,
        wspdname: attrs.WSPDNAME,
        wdirname: attrs.WDIRNAME
    }
    dset = open_netcdf_or_zarr(
        filename_or_fileglob=filename_or_fileglob,
        file_format=file_format,
        chunks=chunks,
        mapping=mapping,
    )
    # dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks, combine="by_coords")
    _units = dset[attrs.SPECNAME].attrs.get("units", "")
    _variable_name = attrs.SPECNAME
    dset = dset.rename({k: v for k, v in mapping.items() if k in dset})
    dset[attrs.SPECNAME].attrs.update(
        {"_units": _units, "_variable_name": _variable_name}
    )
    if attrs.DIRNAME not in dset or len(dset.dir) == 1:
        dset[attrs.SPECNAME].attrs.update({"units": "m^{2}.s"})
    set_spec_attributes(dset)
    return dset
