"""
Functions to read data from different file format into SpecDataset objects
"""
import os
import glob
import datetime
import pandas as pd
import xarray as xr
import numpy as np
from collections import OrderedDict
from sortedcontainers import SortedDict, SortedSet
from tqdm import tqdm

from swan import SwanSpecFile, read_tab
from specdataset import SpecDataset
from attributes import *
from misc import uv_to_spddir

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

def read_hotswan(fileglob, dirorder=True):
    """
    Read multiple swan hotfiles into single gridded Dataset
    Input:
        fileglob :: glob pattern specifying multiple hotfiles
        dirorder :: if True ensures directions are sorted
    Output:
        SpecDataset object with different grid parts concateneted
    Remark:
        SWAN hotfiles from mpi runs are split by the number of cores over the largest dim of
        (lat, lon) with overlapping rows or columns that are computed in only one of the split
        hotfiles. Here overlappings are concatenated so that those with higher values are kept
        which assumes non-computed overlapping rows or columns are filled with zeros
    """
    hotfiles = sorted(glob.glob(fileglob))
    assert hotfiles, 'No SWAN file identified with fileglob %s' % (fileglob)

    dsets = [read_swan(hotfiles[0])]
    for hotfile in tqdm(hotfiles[1:]):
        dset = read_swan(hotfile)
        # Ensure we keep non-zeros in overlapping rows or columns 
        overlap = {'lon': set(dsets[-1].lon.values).intersection(dset.lon.values),
                   'lat': set(dsets[-1].lat.values).intersection(dset.lat.values)}
        concat_dim = min(overlap, key=lambda x: len(overlap[x]))
        for concat_val in overlap[concat_dim]:
            slc = {concat_dim: [concat_val]}
            if dsets[-1].efth.loc[slc].sum() > dset.efth.loc[slc].sum():
                dset.efth.loc[slc] = dsets[-1].efth.loc[slc]
            else:
                dsets[-1].efth.loc[slc] = dset.efth.loc[slc]
        dsets.append(dset)
    dset = xr.auto_combine(dsets)
    set_spec_attributes(dset)
    return dset

def read_swans(fileglob, dirorder=True):
    """
    Read multiple swan files into single Dataset
    Input:
        fileglob :: glob pattern specifying multiple files
        dirorder :: if True ensures directions are sorted
    Output:
        SpecDataset object with different sites and cycles concatenated along the 'site' and 'time' dimension
        If multiple cycles are provided, 'time' coordinate is replaced by 'cycletime' multi-index coordinate
    Remarks:
    - Sites are grouped based on filename
    - If multiple sites are provided in fileglob, each site must have same number of cycles
    """
    swans = sorted(glob.glob(fileglob))
    assert swans, 'No SWAN file identified with fileglob %s' % (fileglob)

    sites = SortedSet([os.path.splitext(os.path.basename(f))[0] for f in swans])
    dsets = SortedDict({site: [] for site in sites})

    # Reading in all sites
    for filename in tqdm(swans):
        site = os.path.splitext(os.path.basename(filename))[0]
        dsets[site].append(read_swan(filename, dirorder=dirorder, as_site=True))
    
    # Sanity checking
    time_sizes = [dset.time.size for dset in dsets[site]]
    cycle_sizes = [len(dsets[site]) for site in sites]
    if len(set(time_sizes)) != 1 or len(set(cycle_sizes)) != 1:
        raise IOError('Inconsistent number of time records or cycles among sites')
    
    # Merging into one dataset
    cycles = [dset.time[0].values for dset in dsets[site]]
    dsets = xr.concat([xr.concat(dsets[site], dim=TIMENAME) for site in sites], dim=SITENAME)
    dsets[SITENAME].values = np.arange(len(sites))+1

    # Define multi-index coordinate if reading multiple cycles
    if len(cycles) > 1:
        dsets.rename({'time': 'cycletime'}, inplace=True)
        cycletime = zip([item for sublist in [[c]*t for c,t in zip(cycles, time_sizes)] for item in sublist],
                        dsets.cycletime.values)
        dsets['cycletime'] = pd.MultiIndex.from_tuples(cycletime, names=[CYCLENAME, TIMENAME])
        dsets['cycletime'].attrs = ATTRS[TIMENAME]

    return dsets

def read_swan(filename, dirorder=True, as_site=None):
    """
    Read Spectra off SWAN ASCII file
        - dirorder :: If True reorder spectra read from file so that directions are sorted
        - as_site :: If true enforces SpecArray is of 'site' type (1D site coordinate defines location)
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
    tab = None
    
    if as_site:
        swanfile.is_grid = False

    spec_list = [s for s in swanfile.readall()]

    if swanfile.is_tab:
        try:
            tab = read_tab(swanfile.tabfile)
            if len(swanfile.times) == tab.index.size:
                if 'X-wsp' in tab and 'Y-wsp' in tab:
                    tab['wspd'], tab['wdir'] = uv_to_spddir(tab['X-wsp'], tab['Y-wsp'], coming_from=True)
            else:
                print "Warning: times in %s and %s not consistent, not appending winds and depth" % (
                    swanfile.filename, swanfile.tabfile)
                tab = None
        except Exception as exc:
            print "Cannot parse depth and winds from %s:\n%s" % (swanfile.tabfile, exc)

    if swanfile.is_grid:
        lons = sorted(np.unique(lons))
        lats = sorted(np.unique(lats))
        arr = np.array(spec_list).reshape(len(times), len(lons), len(lats), len(freqs), len(dirs))
        dset = xr.DataArray(
            data=np.swapaxes(arr, 1, 2),
            coords=OrderedDict(((TIMENAME, times), (LATNAME, lats), (LONNAME, lons), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, LATNAME, LONNAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            ).to_dataset()

        if tab is not None and 'wspd' in tab:
            dset[WSPDNAME] = xr.DataArray(data=tab['wspd'].values.reshape(-1,1,1), dims=[TIMENAME, LATNAME, LONNAME])
            dset[WDIRNAME] = xr.DataArray(data=tab['wdir'].values.reshape(-1,1,1), dims=[TIMENAME, LATNAME, LONNAME])
        if tab is not None and 'dep' in tab:
            dset[DEPNAME] = xr.DataArray(data=tab['dep'].values.reshape(-1,1,1), dims=[TIMENAME, LATNAME, LONNAME])
    else:
        arr = np.array(spec_list).reshape(len(times), len(sites), len(freqs), len(dirs))
        dset = xr.DataArray(
            data=arr,
            coords=OrderedDict(((TIMENAME, times), (SITENAME, sites), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, SITENAME, FREQNAME, DIRNAME),
            name=SPECNAME,
        ).to_dataset()

        if tab is not None and 'wspd' in tab:
            dset[WSPDNAME] = xr.DataArray(data=tab['wspd'].values.reshape(-1,1), dims=[TIMENAME, SITENAME])
            dset[WDIRNAME] = xr.DataArray(data=tab['wdir'].values.reshape(-1,1), dims=[TIMENAME, SITENAME])
        if tab is not None and 'dep' in tab:
            dset[DEPNAME] = xr.DataArray(data=tab['dep'].values.reshape(-1,1), dims=[TIMENAME, SITENAME])

        dset[LATNAME] = xr.DataArray(data=lats, coords={SITENAME: sites}, dims=[SITENAME])
        dset[LONNAME] = xr.DataArray(data=lons, coords={SITENAME: sites}, dims=[SITENAME])

    set_spec_attributes(dset)
    return dset

def read_octopus(filename):
    raise NotImplementedError('No Octopus read function defined')

def read_json(self,filename):
    from cfjson.xrdataset import CFJSONinterface
    raise NotImplementedError('Cannot read CFJSON format')

if __name__ == '__main__':

    import datetime
    import matplotlib.pyplot as plt

    # fileglob = '/source/pyspectra/tests/swan/hot/aklislr.20170412_00z.hot-???'
    fileglob = '/source/pyspectra/tests/swan/hot/aklishr.20170412_12z.hot-???'
    ds = read_hotswan(fileglob)
    plt.figure()
    ds.spec.hs().plot(cmap='jet')
    plt.show()

    # fileglob = '/source/pyspectra/tests/swan/swn*/*.spec'

    # t0 = datetime.datetime.now()
    # ds = read_swans(fileglob, dirorder=True)
    # print (datetime.datetime.now()-t0).total_seconds()

    # fileglob = '/source/pyspectra/tests/swan/swn20170407_12z/aucki.spec'
    # ds = read_swans(fileglob, dirorder=True)

    # fileglob = '/source/pyspectra/tests/swan/swn20170407_12z/*.spec'
    # ds = read_swans(fileglob, dirorder=True)

