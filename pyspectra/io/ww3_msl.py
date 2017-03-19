import xarray as xr
from ..spectra import SpecArray
from attributes import *

def read_ww3_msl(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in MSL netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'site']
    Returns:
    - dset :: Dataset with spec accessor class attached to spectra DataArray
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    dset.rename({'freq': FREQNAME, 'dir': DIRNAME}, inplace=True)#, 'SITE': SITENAME})
    dset[SPECNAME] = (dset['specden'].astype('float32')+127.) * dset['factor']
    dset[SPECNAME].attrs = SPECATTRS
    dset = dset.drop(['specden','factor', 'df'])
    return SpecArray(dset)

def to_ww3(filename):
    raise NotImplementedError('Cannot write to native WW3 format')