"""
Functions to read data from different file format into SpecDataset objects
"""
import os
import glob
import xarray as xr
import numpy as np
from sortedcontainers import SortedDict, SortedSet
from tqdm import tqdm

from cfjson.xrdataset import *

from swan import SwanSpecFile
from specdataset import SpecDataset
from attributes import *

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
    return dset

def read_ww3(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in native netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documentation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'station']
    Returns:
    - dset :: SpecDataset instance
    """    
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    dset.rename({'frequency': FREQNAME, 'direction': DIRNAME, 'station': SITENAME, 'efth': SPECNAME,
        'longitude': LONNAME, 'latitude': LATNAME}, inplace=True)
    dset[SPECNAME].values = np.radians(dset[SPECNAME].values)
    set_spec_attributes(dset)
    return dset

def read_ww3_msl(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in MSL netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documentation)
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

def read_swans(fileglob, dirorder=True):
    """
    Read multiple swan files into single Dataset
        - fileglob :: glob pattern specifying multiple files
        - dirorder :: if True ensures directions are sorted
    Assumes filenames are the same for the same location
    """
    swans = sorted(glob.glob(fileglob))
    assert swans, 'No SWAN file identified with fileglob %s' % (fileglob)

    sites = SortedSet([os.path.splitext(os.path.basename(f))[0] for f in swans])
    dsets = SortedDict({site: [] for site in sites})

    for filename in tqdm(swans):
        site = os.path.splitext(os.path.basename(filename))[0]
        dsets[site].append(read_swan(filename, dirorder=True))


    # import ipdb; ipdb.set_trace()
    return dsets

def read_swan(filename, dirorder=True, cycle=False):
    """
    Read Spectra off SWAN ASCII file
    - dirorder :: If True reorder spectra read from file so that directions are sorted
    - cycle :: If True defines cycle dimension in Dataset
    Returns:
    - dset :: SpecDataset instance
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

    if cycle:  
        pass

    set_spec_attributes(dset)
    return dset

def read_octopus(filename):
    raise NotImplementedError('No Octopus read function defined')

def read_json(self,filename):
    raise NotImplementedError('Cannot read CFJSON format')

def expand_cycle(dset):
    """
    Expand data array to include cycle dimension
        - Input :: xarray Dataset
        - Output :: same Dataset extended with 'cycle' dimension of length 1
    Value for cycle coordinate is taken from first value in time coordinate
    """
    dset_out = xr.Dataset()
    cycle_time = dset['time'][0].values
    cycle_coord = xr.DataArray(data=[cycle_time], coords={CYCLENAME: [cycle_time]}, dims=(CYCLENAME,), name=CYCLENAME)

    for dvar in dset.data_vars:
        coords = SortedDict({CYCLENAME: cycle_coord})
        coords.update(dset[dvar].coords)
        dset_out[dvar] = xr.DataArray(data=dset[dvar].values[np.newaxis,...],
                                      coords=coords,
                                      dims=[CYCLENAME]+list(dset[dvar].dims),
                                      name=dvar,
                                      attrs=dset[dvar].attrs)
    set_spec_attributes(dset_out)
    return dset_out

if __name__ == '__main__':

    ds = read_swans('/source/pyspectra/tests/swan/swn*/*.spec', dirorder=True)