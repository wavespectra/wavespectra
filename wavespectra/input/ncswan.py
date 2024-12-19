"""Read Native SWAN netCDF spectra files."""

from xarray.backends import BackendEntrypoint
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import uv_to_spddir
from wavespectra.input import open_netcdf_or_zarr, to_keep

R2D = 180 / np.pi

MAPPING = {
    "time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "points": attrs.SITENAME,
    "density": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "depth": attrs.DEPNAME,
}


def read_ncswan(filename_or_fileglob, file_format="netcdf", mapping=MAPPING, chunks={}):
    """Read Spectra from SWAN native netCDF format.

    Args:
        - filename_or_fileglob (str, list, fileobj): filename, fileglob specifying multiple
          files, or a file object to read.
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
    return from_ncswan(dset)


def from_ncswan(dset):
    """Format SWAN netcdf dataset to receive wavespectra accessor.

    Args:
        - dset (xr.Dataset): Dataset created from a SWAN netcdf file.

    Returns:
        - Formated dataset with the SpecDataset accessor in the `spec` namespace.

    """
    vars_and_dims = set(dset.data_vars) | set(dset.dims)
    mapping = {k: v for k, v in MAPPING.items() if k != v and k in vars_and_dims}
    dset = dset.rename(mapping)
    # Ensuring lon,lat are not function of time
    if attrs.LONNAME in dset and attrs.TIMENAME in dset[attrs.LONNAME].dims:
        dset[attrs.LONNAME] = dset[attrs.LONNAME].isel(drop=True, **{attrs.TIMENAME: 0})
    if attrs.LATNAME in dset and attrs.TIMENAME in dset[attrs.LATNAME].dims:
        dset[attrs.LATNAME] = dset[attrs.LATNAME].isel(drop=True, **{attrs.TIMENAME: 0})

    # Calculating wind speeds and directions
    if "xwnd" in dset and "ywnd" in dset:
        dset[attrs.WSPDNAME], dset[attrs.WDIRNAME] = uv_to_spddir(
            dset["xwnd"], dset["ywnd"], coming_from=True
        )
    # Only selected variables to be returned
    to_drop = list(set(dset.data_vars.keys()) - to_keep)
    # Converting from radians
    dset[attrs.SPECNAME] /= R2D
    if attrs.DIRNAME in dset:
        dset = dset.assign_coords({attrs.DIRNAME: (dset[attrs.DIRNAME] * R2D) % 360})
    # Ensure site is a coordinate
    if attrs.SITENAME in dset.dims and attrs.SITENAME not in dset.coords:
        dset[attrs.SITENAME] = np.arange(1, len(dset[attrs.SITENAME]) + 1)
    # Setting standard attributes
    set_spec_attributes(dset)
    return dset.drop_vars(to_drop)


class NCSwanBackendEntrypoint(BackendEntrypoint):
    """Swan netcdf backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        file_format="netcdf",
        mapping=MAPPING,
    ):
        return read_ncswan(filename_or_obj, file_format=file_format, mapping=mapping)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open SWAN netcdf or zarr spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
