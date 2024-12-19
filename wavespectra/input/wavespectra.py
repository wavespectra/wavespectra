"""Read netCDF or ZARR formatted with wavespectra conventions."""

from xarray.backends import BackendEntrypoint

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import set_spec_attributes
from wavespectra.input import open_netcdf_or_zarr


def read_wavespectra(filename_or_fileglob, file_format="netcdf", chunks={}):
    """Read Spectra from from netCDF or ZARR format in Wavespectra convention.

    Args:
        - filename_or_fileglob (str, list, fileobj): filename, fileglob specifying
          multiple files, or a file object to read.
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
    dset = open_netcdf_or_zarr(
        filename_or_fileglob=filename_or_fileglob,
        file_format=file_format,
        chunks=chunks,
    )
    set_spec_attributes(dset)
    return dset


class WavespectraBackendEntrypoint(BackendEntrypoint):
    """Wavespectra backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        file_format="netcdf",
    ):
        return read_wavespectra(filename_or_obj, file_format=file_format)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open Wavespectra spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
