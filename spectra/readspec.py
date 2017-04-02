"""
Functions to read data from different file format into SpecDataset objects
"""
import xarray as xr
import numpy as np

from cfjson.xrdataset import *

from pyspectra.spectra.specdataset import SpecDataset
from pyspectra.spectra.io.attributes import *

def read_netcdf(filename_or_fileglob,
                chunks={},
                freqname=FREQNAME,
                dirname=DIRNAME,
                sitename=SITENAME,
                specname=SPECNAME,
                lonname=LONNAME,
                latname=LATNAME):
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
    set_spec_attributes(dset)
    return SpecDataset(dset)

def read_ww3(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in native netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'station']
    Returns:
    - dset :: SpecDataset instance
    """    
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    dset.rename({'frequency': FREQNAME, 'direction': DIRNAME, 'station': SITENAME, 'efth': SPECNAME,
        'longitude': LONNAME, 'latitude': LATNAME}, inplace=True)
    dset[SPECNAME].values = np.radians(dset[SPECNAME].values)
    set_spec_attributes(dset)
    return SpecDataset(dset)

def read_ww3_msl(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in MSL netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'site']
    Returns:
    - dset :: SpecDataset instance
    """    
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    dset.rename({'freq': FREQNAME, 'dir': DIRNAME}, inplace=True)#, 'SITE': SITENAME})
    dset[SPECNAME] = (dset['specden'].astype('float32')+127.) * dset['factor']
    dset = dset.drop(['specden','factor', 'df'])
    set_spec_attributes(dset)
    return dset

def read_swan(filename, dirorder=True):
    """
    Read Spectra off SWAN ASCII file
    - dirorder :: If True reorder spectra read from file so that directions are sorted
    Returns:
    - dset :: SpecArray
    """
    swanfile = SwanSpecFile(filename, dirorder=dirorder)
    times = swanfile.times
    lons = swanfile.x
    lats = swanfile.y
    sites = np.arange(len(lons))+1
    freqs = swanfile.freqs
    dirs = swanfile.dirs
    
    spec_list = swanfile.readall()
    if swanfile.is_grid:
        # Looks like gridded data, grid DataArray accordingly
        arr = np.array([s for s in spec_list]).reshape(len(times), len(lons), len(lats), len(freqs), len(dirs))
        dset = xr.DataArray(
            data=np.swapaxes(arr, 1, 2),
            coords=OrderedDict(((TIMENAME, times), (LATNAME, lats), (LONNAME, lons), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, LATNAME, LONNAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            ).to_dataset()
    else:
        # Keep it with sites dimension
        arr = np.array([s for s in spec_list]).reshape(len(times), len(sites), len(freqs), len(dirs))
        dset = xr.DataArray(
            arr,
            coords=OrderedDict(((TIMENAME, times), (SITENAME, sites), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, SITENAME, FREQNAME, DIRNAME),
            name=SPECNAME,
        ).to_dataset()
        dset[LATNAME] = xr.DataArray(data=lats, coords={SITENAME: sites}, dims=[SITENAME])
        dset[LONNAME] = xr.DataArray(data=lons, coords={SITENAME: sites}, dims=[SITENAME])
    set_spec_attributes(dset)
    return SpecDataset(dset)

def read_octopus(filename):
    raise NotImplementedError('No Octopus read function defined')

def read_json(self,filename):
    raise NotImplementedError('Cannot read CFJSON format')