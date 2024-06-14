"""Read NDBC netCDF spectra files.

https://dods.ndbc.noaa.gov/

"""
import logging
from xarray.backends import BackendEntrypoint
import xarray as xr
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import D2R


logger = logging.getLogger(__name__)

MAPPING = {
    "time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "spectral_wave_density": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "depth": attrs.DEPNAME,
}


def read_ndbc(url, directional=True, dd=10.0, chunks={}):
    """Read Spectra from NDBC netCDF format.

    Args:
        - url (str): Thredds URL or local path of file or fileobj to read.
        - directional (bool): Constructs 2D spectra if True, returns 1D if False.
        - dd (float): Directional resolution for 2D spectra (deg).
        - chunks (dict): Chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions.

    Returns:
        - dset (SpecDataset): spectra dataset object read from NDBC file.

    Note:
        - Any missing values within directional variables in the NDBC dataset will
          result in NaN in the directional spectra for the respective timestamps.
          These spectra can only be read as 1D by setting `directional=False`.
        - If file is large to fit in memory, consider specifying chunks for
          'time' or other non-spectral dims.

    """
    dset = xr.open_dataset(url).chunk(chunks)
    return from_ndbc(dset, directional=directional, dd=dd)


def _construct_spectra(ef, swd1, swd2, swr1, swr2, dir):
    """Construct 2D spectra."""
    d = 0.5 + swr1 * np.cos(D2R * (dir - swd1)) + swr2 * np.cos(2 * D2R * (dir - swd2))
    return ef * d * D2R / np.pi


def from_ndbc(dset, directional=True, dd=10.0):
    """Format NDBC netcdf dataset to implement the wavespectra accessor.

    Args:
        - dset (xr.Dataset): Dataset created from a NDBC netcdf file.
        - directional (bool): Constructs 2D spectra if True, returns 1D if False.
        - dd (float): Directional resolution for 2D spectra (deg).

    Returns:
        - Formated dataset with the SpecDataset accessor in the `spec` namespace.

    Note:
        - If any of the missing variables are not available in the dataset, the
          function will return the dataset as 1D spectra even if `directional=True`.

    """
    # Some datasets have different naming convention
    mapping = {}
    if "waveFrequency" in dset.dims:
        mapping.update({"waveFrequency": "frequency"})
    if "waveTime" in dset.dims:
        mapping.update({"waveTime": "time"})
    if "waveEnergyDensity" in dset.data_vars:
        mapping.update({"waveEnergyDensity": "spectral_wave_density"})
    if "gpsLatitude" in dset.data_vars:
        mapping.update({"gpsLatitude": "latitude"})
    if "gpsLongitude" in dset.data_vars:
        mapping.update({"gpsLongitude": "longitude"})
    dset = dset.rename(mapping)

    # Force it to read as nondirectional spectra if directional data isn't available
    v = {"mean_wave_dir", "principal_wave_dir", "wave_spectrum_r1", "wave_spectrum_r2"}
    if directional is True and v - dset.data_vars.keys():
        logger.warning("Directional data not available. Reading as 1D spectra.")
        directional = False

    dset = dset.transpose(..., "frequency")
    if directional:
        dirs = np.arange(0, 360, dd)
        dset = _construct_spectra(
            ef=dset.spectral_wave_density,
            swd1=dset.mean_wave_dir,
            swd2=dset.principal_wave_dir,
            swr1=dset.wave_spectrum_r1,
            swr2=dset.wave_spectrum_r2,
            dir=xr.DataArray(dirs, dims=(attrs.DIRNAME,), coords={attrs.DIRNAME: dirs}),
        )
    else:
        dset = dset.spectral_wave_density
    dset = dset.to_dataset(name="efth")

    vars_and_dims = set(dset.data_vars) | set(dset.dims)
    mapping = {k: v for k, v in MAPPING.items() if k != v and k in vars_and_dims}
    dset = dset.rename(mapping)

    set_spec_attributes(dset)

    return dset


class NDBCBackendEntrypoint(BackendEntrypoint):
    """NDBC netcdf backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        directional=True,
        dd=10.0,
    ):
        return read_ndbc(filename_or_obj, directional=directional, dd=dd)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open NDBC netcdf spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
