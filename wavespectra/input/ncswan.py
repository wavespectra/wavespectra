"""Read Native SWAN netCDF spectra files."""
import xarray as xr
import numpy as np
import dask.array as da

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.misc import uv_to_spddir

R2D = 180 / np.pi


def read_ncswan(filename_or_fileglob, chunks={}, file_format="netcdf"):
    """Read Spectra from SWAN native netCDF format.

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
    return from_ncswan(dset)


def from_ncswan(dset):
    """Format SWAN netcdf dataset to receive wavespectra accessor.

    Args:
        dset (xr.Dataset): Dataset created from a SWAN netcdf file.

    Returns:
        Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """
    _units = dset.density.attrs.get("units", "")
    dset = dset.rename(
        {
            "frequency": attrs.FREQNAME,
            "direction": attrs.DIRNAME,
            "points": attrs.SITENAME,
            "density": attrs.SPECNAME,
            "longitude": attrs.LONNAME,
            "latitude": attrs.LATNAME,
            "depth": attrs.DEPNAME,
        }
    )
    # Ensuring lon,lat are not function of time
    if attrs.TIMENAME in dset[attrs.LONNAME].dims:
        dset[attrs.LONNAME] = dset[attrs.LONNAME].isel(drop=True, **{attrs.TIMENAME: 0})
        dset[attrs.LATNAME] = dset[attrs.LATNAME].isel(drop=True, **{attrs.TIMENAME: 0})
    # Calculating wind speeds and directions
    if "xwnd" in dset and "ywnd" in dset:
        dset[attrs.WSPDNAME], dset[attrs.WDIRNAME] = uv_to_spddir(
            dset["xwnd"], dset["ywnd"], coming_from=True
        )
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
    # Setting standard names and storing original file attributes
    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update(
        {"_units": _units, "_variable_name": attrs.SPECNAME}
    )
    # Converting from radians
    dset[attrs.SPECNAME] /= R2D
    if attrs.DIRNAME in dset:
        dset[attrs.DIRNAME] *= R2D
        dset[attrs.DIRNAME] %= 360
        # dset = dset.sortby(attrs.DIRNAME)
    # Adjustting attributes if 1D
    if attrs.DIRNAME not in dset or len(dset.dir) == 1:
        dset[attrs.SPECNAME].attrs.update({"units": "m^{2}.s"})
    # Ensure site is a coordinate
    if attrs.SITENAME in dset.dims and attrs.SITENAME not in dset.coords:
        dset[attrs.SITENAME] = np.arange(1, len(dset[attrs.SITENAME]) + 1)
    return dset.drop(to_drop)


if __name__ == "__main__":
    import os

    FILES_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../tests/sample_files"
    )
    ds_spec = read_ncswan(os.path.join(FILES_DIR, "swanfile.nc"))
    ds_swan = xr.open_dataset(os.path.join(FILES_DIR, "swanfile.nc"))
