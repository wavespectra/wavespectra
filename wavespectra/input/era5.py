"""Read ERA5 2D Wave Spectra NetCDF files"""
import numpy as np

from wavespectra.input.netcdf import read_netcdf


def read_era5(filename_or_fileglob, chunks={}, freqs=None, dirs=None):
    """Read Spectra from ECMWF ERA5 netCDF format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - freqs (list): list of frequencies. By default use all 30 ERA5 frequencies.
        - dirs (list): list of directions. By default use all 24 ERA5 directions.

    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file.

    Note:
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    default_freqs = np.full(30, 0.03453) * (1.1 ** np.arange(0, 30))
    default_dirs = direction = np.arange(7.5, 352.5 + 15, 15)

    dset = read_netcdf(
        filename_or_fileglob,
        specname="d2fd",
        freqname="frequency",
        dirname="direction",
        lonname="longitude",
        latname="latitude",
        timename="time",
        chunks=chunks,
    )

    # Convert ERA5 format to wavespectra format
    dset = 10 ** dset
    dset = dset.fillna(0)

    dset["freq"] = freqs if freqs else default_freqs
    dset["dir"] = dirs if dirs else default_dirs

    return dset
