#Generic netcdf in/output
import xarray as xr
from attributes import *

def read_netcdf(filename_or_fileglob,
                chunks={},
                freqname=FREQNAME,
                dirname=DIRNAME,
                sitename=SITENAME,
                specname=SPECNAME,
                lonname=LONNAME,
                latname=LATNAME):
    from spectra import SpecDataset
    """
    Read Spectra off generic netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'station']
    Returns:
    - dset :: SpecDataset instance
    -----
    Note that this current assumes frequency as Hz, direction as degrees and spectral energy as m^2s
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    if sitename in dset.dims: #Sites based
        dset.rename({freqname: FREQNAME, dirname: DIRNAME, sitename: SITENAME, specname: SPECNAME}, inplace=True)
    else: #Gridded
        dset.rename({freqname: FREQNAME, dirname: DIRNAME, lonname: LONNAME, latname: LATNAME, specname: SPECNAME}, inplace=True)
    dset[SPECNAME].attrs = SPECATTRS
    return SpecDataset(dset)

#xarray already provides this
#def to_nc(filename):
#    raise NotImplementedError('Cannot write to native WW3 format')