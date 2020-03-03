"""Read ERA5 2D Wave Spectra NetCDF files"""
from wavespectra.input.netcdf import read_netcdf

def read_era5(filename_or_fileglob, chunks={}, freqs=None, dirs=None):

    """Read Spectra from ECMWF ERA5 netCDF format.
    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - freqs (list): list of frequencies. By default use all 30 ERA5 frequencies
        - dirs (list): list of directions. By default use all 24 ERA5 directions
    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file
    Note:
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.
    """
    
    default_freqs = [
        0.03453   , 0.037983  , 0.0417813 , 0.04595943, 0.05055537,
        0.05561091, 0.061172  , 0.0672892 , 0.07401812, 0.08141993,
        0.08956193, 0.09851812, 0.10836993, 0.11920693, 0.13112762,
        0.14424038, 0.15866442, 0.17453086, 0.19198394, 0.21118234,
        0.23230057, 0.25553063, 0.28108369, 0.30919206, 0.34011127,
        0.3741224 , 0.41153464, 0.4526881 , 0.49795691, 0.5477526
    ]
    
    default_dirs = [
        7.5  , 22.5 , 37.5 , 52.5 , 67.5 , 82.5 , 97.5 , 112.5,
        127.5, 142.5, 157.5, 172.5, 187.5, 202.5, 217.5, 232.5,
        247.5, 262.5, 277.5, 292.5, 307.5, 322.5, 337.5, 352.5
    ]
    
    dset = read_netcdf(
        filename_or_fileglob,
        specname='d2fd',
        freqname='frequency',
        dirname='direction',
        lonname='longitude',
        latname='latitude',
        timename='time',
        chunks=chunks
    )
    
    # Convert ERA5 format to wavespectra format
    dset = 10 ** dset
    dset = dset.fillna(0)

    dset['freq'] = freqs if freqs else default_freqs
    dset['dir'] = dirs if dirs else default_dirs
    
    return dset
