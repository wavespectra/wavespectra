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

from swan import SwanSpecFile, read_tab, interp_spec
from specdataset import SpecDataset
from attributes import *
from misc import uv_to_spddir, to_datetime, flatten_list

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
    dset.rename({'freq': FREQNAME, 'dir': DIRNAME, 'wsp': WSPDNAME}, inplace=True)#, 'SITE': SITENAME})
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

def read_swans2(fileglob, ndays=None, int_freq=True, int_dir=True, dirorder=True):
    """
    Read multiple swan files into single Dataset
    Input:
        fileglob :: glob pattern specifying multiple files
        ndays :: maximum number of days from each site to keep, choose None to keep whole time periods
        int_freq :: {array, True, False} frequency array for interpolating onto
            - array: 1d array specifying frequencies to interpolate onto
            - True: logarithm array is constructed so fmin=0.04180 Hz, fmax=0.71856 Hz, df=0.1f
            - False: No interpolation performed in the frequency space (keep it from original spectrum)
        int_dir :: {array, True, False} direction array for interpolating onto
            - array: 1d array specifying directions to interpolate onto
            - True: circular array is constructed so dd=10 degrees
            - False: No interpolation performed in the direction space (keep it from original spectrum)
        dirorder :: if True ensures directions are sorted
    Output:
        SpecDataset object with different sites and cycles concatenated along the 'site' and 'time' dimension
        If multiple cycles are provided, 'time' coordinate is replaced by 'cycletime' multi-index coordinate
    Remarks:
    - If more than one cycle is prescribed from fileglob, each cycle must have same number of sites
    - If multiple sites are provided in fileglob, each site must have same number of cycles
    """
    swans = sorted(glob.glob(fileglob))
    assert swans, 'No SWAN file identified with fileglob %s' % (fileglob)

    # Default spectral basis for interpolating
    if int_freq == True:
        int_freq = [0.04118 * 1.1**n for n in range(31)]
    elif int_freq == False:
        int_freq = None
    if int_dir == True:
        int_dir = np.arange(0, 360, 10)
    elif int_dir == False:
        int_dir = None

    cycles = list()
    dsets = SortedDict()

    for filename in tqdm(swans):
        swanfile = SwanSpecFile(filename, dirorder=dirorder)
        times = swanfile.times
        lons = swanfile.x
        lats = swanfile.y
        sites = [os.path.splitext(os.path.basename(filename))[0]] if len(lons)==1 else np.arange(len(lons))+1
        freqs = swanfile.freqs
        dirs = swanfile.dirs
        tab = None
        swanfile.is_grid = False

        # spec_list = flatten_list([s for s in swanfile.readall()], [])
        spec_list = [s for s in swanfile.readall()]
        if ndays is not None:
            tend = times[0] + datetime.timedelta(days=ndays)
            iend = times.index(min(times, key=lambda d: abs(d - tend)))
            times = times[0:iend]
            spec_list = spec_list[0:iend]
        spec_list = flatten_list(spec_list, [])

        if int_freq is not None or int_dir is not None:
            spec_list = [interp_spec(spec, freqs, dirs, int_freq, int_dir) for spec in spec_list]
            freqs = int_freq if int_freq is not None else freqs
            dirs = int_dir if int_dir is not None else dirs

        try:
            arr = np.array(spec_list).reshape(len(times), len(sites), len(freqs), len(dirs))
            cycle = times[0]
            if cycle not in dsets:
                dsets[cycle] = arr
                nsites = 1
            else:
                dsets[cycle] = np.concatenate((dsets[cycle], arr), axis=1)
                nsites += 1
        except:
            # raise
            import ipdb; ipdb.set_trace()







    # # Read in each file
    # dsets = SortedDict()
    # cycles = list()
    # for filename in tqdm(swans):
    #     dset = read_swan(filename, dirorder=dirorder, as_site=True)
    #     cycle = dset.time[0].values
    #     if cycle not in dsets:
    #         dsets[cycle] = dset
    #         cycles.append(cycle)
    #     else:
    #         dsets[cycle] = xr.concat([dsets[cycle], dset], dim=SITENAME)#, data_vars='minimal')

    # # Sanity checking
    # time_sizes = [dset.time.size for dset in dsets.values()]
    # site_sizes = [dset.site.size for dset in dsets.values()]
    # if len(set(site_sizes))!=1:
    #     raise IOError('Inconsistent number of sites over different cycles')

    # dset_out = dsets[cycles[0]].copy(deep=True)
    # dsets.pop(cycles[0])

    # # Cycles merging
    # if dsets:
    #     for cycle,dset in tqdm(dsets.items()):
    #         dset_out = xr.concat([dset_out, dset], dim=TIMENAME, data_vars='minimal')
    #         dsets.pop(cycle)
    #     # Define multi-index coordinate
    #     dset_out.rename({'time': 'cycletime'}, inplace=True)
    #     cycletime = zip([item for sublist in [[c]*t for c,t in zip(cycles, time_sizes)] for item in sublist],
    #                     dset_out.cycletime.values)
    #     dset_out['cycletime'] = pd.MultiIndex.from_tuples(cycletime, names=[CYCLENAME, TIMENAME])
    #     dset_out['cycletime'].attrs = ATTRS[TIMENAME]
    
    return dsets

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
    - If more than one cycle is prescribed from fileglob, each cycle must have same number of sites
    - 
    - If multiple sites are provided in fileglob, each site must have same number of cycles
    """
    swans = sorted(glob.glob(fileglob))
    assert swans, 'No SWAN file identified with fileglob %s' % (fileglob)

    # Read in each file
    dsets = SortedDict()
    cycles = list()
    for filename in tqdm(swans):
        dset = read_swan(filename, dirorder=dirorder, as_site=True)
        if dset.time.size==181 and dset.freq.size==24 and dset.dir.size==36:
            cycle = dset.time[0].values
            if cycle not in dsets:
                dsets[cycle] = dset
                cycles.append(cycle)
            else:
                dsets[cycle] = xr.concat([dsets[cycle], dset], dim=SITENAME)#, data_vars='minimal')
        else:
            print "Different dimensions: %s" % (filename)

    # Sanity checking
    time_sizes = [dset.time.size for dset in dsets.values()]
    site_sizes = [dset.site.size for dset in dsets.values()]
    if len(set(site_sizes))!=1:
        raise IOError('Inconsistent number of sites over different cycles')

    dset_out = dsets[cycles[0]].copy(deep=True)
    dsets.pop(cycles[0])

    # Cycles merging
    if dsets:
        for cycle,dset in tqdm(dsets.items()):
            dset_out = xr.concat([dset_out, dset], dim=TIMENAME, data_vars='minimal')
            dsets.pop(cycle)
        # Define multi-index coordinate
        dset_out.rename({'time': 'cycletime'}, inplace=True)
        cycletime = zip([item for sublist in [[c]*t for c,t in zip(cycles, time_sizes)] for item in sublist],
                        dset_out.cycletime.values)
        dset_out['cycletime'] = pd.MultiIndex.from_tuples(cycletime, names=[CYCLENAME, TIMENAME])
        dset_out['cycletime'].attrs = ATTRS[TIMENAME]
    
    return dset_out

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
    sites = [os.path.splitext(os.path.basename(filename))[0]] if len(lons)==1 else np.arange(len(lons))+1
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
                    tab[WSPDNAME], tab[WDIRNAME] = uv_to_spddir(tab['X-wsp'], tab['Y-wsp'], coming_from=True)
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

        if tab is not None and WSPDNAME in tab:
            dset[WSPDNAME] = xr.DataArray(data=tab[WSPDNAME].values.reshape(-1,1,1), dims=[TIMENAME, LATNAME, LONNAME])
            dset[WDIRNAME] = xr.DataArray(data=tab[WDIRNAME].values.reshape(-1,1,1), dims=[TIMENAME, LATNAME, LONNAME])
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

        if tab is not None and WSPDNAME in tab:
            dset[WSPDNAME] = xr.DataArray(data=tab[WSPDNAME].values.reshape(-1,1), dims=[TIMENAME, SITENAME])
            dset[WDIRNAME] = xr.DataArray(data=tab[WDIRNAME].values.reshape(-1,1), dims=[TIMENAME, SITENAME])
        if tab is not None and 'dep' in tab:
            dset[DEPNAME] = xr.DataArray(data=tab['dep'].values.reshape(-1,1), dims=[TIMENAME, SITENAME])

        dset[LATNAME] = xr.DataArray(data=lats, coords={SITENAME: sites}, dims=[SITENAME])
        dset[LONNAME] = xr.DataArray(data=lons, coords={SITENAME: sites}, dims=[SITENAME])

    set_spec_attributes(dset)
    return dset

def read_octopus(filename):
    raise NotImplementedError('No Octopus read function defined')

def read_json(self,filename):
    raise NotImplementedError('Cannot read CFJSON format')

if __name__ == '__main__':

    import datetime
    import matplotlib.pyplot as plt

    # fileglob = '/source/pyspectra/tests/swan/hot/aklislr.20170412_00z.hot-???'
    # fileglob = '/source/pyspectra/tests/swan/hot/aklishr.20170412_12z.hot-???'
    # ds = read_hotswan(fileglob)
    # plt.figure()
    # ds.spec.hs().plot(cmap='jet')
    # plt.show()

    fileglob = '/source/pyspectra/tests/swan/swn*/*.spec'

    t0 = datetime.datetime.now()
    ds = read_swans(fileglob, dirorder=True)
    print (datetime.datetime.now()-t0).total_seconds()

    # fileglob = '/source/pyspectra/tests/swan/swn20170407_12z/aucki.spec'
    # ds = read_swans(fileglob, dirorder=True)

    # fileglob = '/source/pyspectra/tests/swan/swn20170407_12z/*.spec'
    # ds = read_swans(fileglob, dirorder=True)

