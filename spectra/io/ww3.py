import xarray as xr
import numpy as np
from spectra.specarray import SpecArray
from spectra.io.attributes import *

def read_ww3(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in native netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'station']
    Returns:
    - dset :: Dataset with spec accessor class attached to spectra DataArray
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    dset.rename({'frequency': FREQNAME, 'direction': DIRNAME, 'station': SITENAME, 'efth': SPECNAME}, inplace=True)
    dset[SPECNAME].values = np.radians(dset[SPECNAME].values)
    set_spec_attributes(dset)
    return dset

def to_ww3(filename):
    raise NotImplementedError('Cannot write to native WW3 format')