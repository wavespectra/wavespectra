"""Read generic netCDF spectra files."""
from xarray.backends import BackendEntrypoint

from wavespectra.specdataset import SpecDataset
from wavespectra.input import open_netcdf
from wavespectra.core.attributes import attrs, set_spec_attributes


def read_netcdf(
    filename_or_fileglob,
    chunks={},
    freqname=attrs.FREQNAME,
    dirname=attrs.DIRNAME,
    sitename=attrs.SITENAME,
    specname=attrs.SPECNAME,
    lonname=attrs.LONNAME,
    latname=attrs.LATNAME,
    timename=attrs.TIMENAME,
):
    """Read Spectra from generic netCDF format.

    Args:
        - filename_or_fileglob (str, list, filelike): filename, fileglob specifying multiple
          files, or filelike object to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - <coord>name :: coordinate name in netcdf, used for standarising
          dataset.

    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file

    Note:
        - Assumes frequency in :math:`Hz`, direction in :math:`degree` and
          spectral energy in :math:`m^{2}degree^{-1}{s}`.
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    dset = open_netcdf(filename_or_fileglob, chunks=chunks)
    coord_map = {
        freqname: attrs.FREQNAME,
        dirname: attrs.DIRNAME,
        lonname: attrs.LONNAME,
        latname: attrs.LATNAME,
        sitename: attrs.SITENAME,
        specname: attrs.SPECNAME,
        timename: attrs.TIMENAME,
    }
    dset = dset.rename({k: v for k, v in coord_map.items() if k in dset})
    set_spec_attributes(dset)
    return dset


class NetCDFBackendEntrypoint(BackendEntrypoint):
    """Netcdf backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        freqname=attrs.FREQNAME,
        dirname=attrs.DIRNAME,
        sitename=attrs.SITENAME,
        specname=attrs.SPECNAME,
        lonname=attrs.LONNAME,
        latname=attrs.LATNAME,
        timename=attrs.TIMENAME,
    ):
        return read_netcdf(
            filename_or_obj,
            freqname=freqname,
            dirname=dirname,
            sitename=sitename,
            specname=specname,
            lonname=lonname,
            latname=latname,
            timename=timename,
        )

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open generic netcdf spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
