"""
Functions to read data from different file format into SpecDataset objects

Commom structure expected to be returned by methods is a SpecDataset with:
- Following variables from attributes.py module should be used to define standarised coordinates:
    - FREQNAME
    - DIRNAME (if 2D)
    - SPECNAME
    - LONNAME (if available)
    - LATNAME (if available)
    - TIMENAME (if available)
- Dataset attributes should be defined before returning using `set_spec_attributes` function
"""
import os
import copy
import glob
import datetime
import pandas as pd
import xarray as xr
import numpy as np
from dateutil import parser
from collections import OrderedDict
from sortedcontainers import SortedDict, SortedSet
from scipy.io import loadmat

from spectra.swan import SwanSpecFile, read_tab
from spectra.specdataset import SpecDataset
import spectra.attributes as attrs
from spectra.misc import uv_to_spddir, to_datetime, interp_spec, flatten_list, to_nautical, dnum_to_datetime

def read_netcdf(filename_or_fileglob,
                chunks={},
                freqname=attrs.FREQNAME,
                dirname=attrs.DIRNAME,
                sitename=attrs.SITENAME,
                specname=attrs.SPECNAME,
                lonname=attrs.LONNAME,
                latname=attrs.LATNAME,
                timename=attrs.TIMENAME):
    """
    Read Spectra off generic netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: dictionary specifying chunk sizes for each dimensions to read as chunked dask array
                By default dataset is loaded using single chunk for all arrays, see xr.open_mfdataset documentation
    - <coord>name :: coordinate name in netcdf for renaming them, by default those defined in attributes.py
    Returns:
    - dset :: SpecDataset instance
    -----
    Assumes frequency in Hz, direction in degrees and spectral energy in m^2/Hz[/degree]
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    _units = dset[specname].attrs.get('units','')
    _variable_name = specname
    coord_map = {freqname: attrs.FREQNAME,
                 dirname: attrs.DIRNAME,
                 lonname: attrs.LONNAME,
                 latname: attrs.LATNAME,
                 sitename: attrs.SITENAME,
                 specname: attrs.SPECNAME,
                 timename: attrs.TIMENAME}
    dset.rename({k:v for k,v in coord_map.items() if k in dset}, inplace=True)
    dset[attrs.SPECNAME].attrs.update({'_units': _units, '_variable_name': _variable_name})
    if 'dir' not in dset or len(dset.dir)==1:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s'})
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
    _units = dset.efth.attrs.get('units','')
    dset.rename({'frequency': attrs.FREQNAME, 'direction': attrs.DIRNAME,
        'station': attrs.SITENAME, 'efth': attrs.SPECNAME, 'longitude': attrs.LONNAME,
        'latitude': attrs.LATNAME, 'wnddir': attrs.WDIRNAME, 'wnd': attrs.WSPDNAME},
        inplace=True)
    if attrs.TIMENAME in dset[attrs.LONNAME].dims:
        dset[attrs.LONNAME] = dset[attrs.LONNAME].isel(drop=True, **{attrs.TIMENAME: 0})
        dset[attrs.LATNAME] = dset[attrs.LATNAME].isel(drop=True, **{attrs.TIMENAME: 0})
    dset[attrs.SPECNAME].values = np.radians(dset[attrs.SPECNAME].values)
    attrs.set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update({'_units': _units, '_variable_name': 'efth'})
    if 'dir' not in dset or len(dset.dir)==1:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s'})
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
    _units = dset.specden.attrs.get('units','')
    dset.rename({'freq': attrs.FREQNAME, 'dir': attrs.DIRNAME, 'wsp': attrs.WSPDNAME}, inplace=True)#, 'SITE': attrs.SITENAME})
    dset[attrs.SPECNAME] = (dset['specden'].astype('float32')+127.) * dset['factor']
    dset = dset.drop(['specden','factor', 'df'])
    attrs.set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update({'_units': _units, '_variable_name': 'specden'})
    if 'dir' not in dset or len(dset.dir)==1:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s'})
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
    for hotfile in hotfiles[1:]:
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
    attrs.set_spec_attributes(dset)
    if 'dir' in dset and len(dset.dir)>1:
        dset[attrs.SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

    return dset

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
                    tab[attrs.WSPDNAME], tab[attrs.WDIRNAME] = uv_to_spddir(tab['X-wsp'], tab['Y-wsp'], coming_from=True)
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
            coords=OrderedDict(((attrs.TIMENAME, times), (attrs.LATNAME, lats), (attrs.LONNAME, lons), (attrs.FREQNAME, freqs), (attrs.DIRNAME, dirs))),
            dims=(attrs.TIMENAME, attrs.LATNAME, attrs.LONNAME, attrs.FREQNAME, attrs.DIRNAME),
            name=attrs.SPECNAME,
            ).to_dataset()

        if tab is not None and attrs.WSPDNAME in tab:
            dset[attrs.WSPDNAME] = xr.DataArray(data=tab[attrs.WSPDNAME].values.reshape(-1,1,1), dims=[attrs.TIMENAME, attrs.LATNAME, attrs.LONNAME])
            dset[attrs.WDIRNAME] = xr.DataArray(data=tab[attrs.WDIRNAME].values.reshape(-1,1,1), dims=[attrs.TIMENAME, attrs.LATNAME, attrs.LONNAME])
        if tab is not None and 'dep' in tab:
            dset[attrs.DEPNAME] = xr.DataArray(data=tab['dep'].values.reshape(-1,1,1), dims=[attrs.TIMENAME, attrs.LATNAME, attrs.LONNAME])
    else:
        arr = np.array(spec_list).reshape(len(times), len(sites), len(freqs), len(dirs))
        dset = xr.DataArray(
            data=arr,
            coords=OrderedDict(((attrs.TIMENAME, times), (attrs.SITENAME, sites), (attrs.FREQNAME, freqs), (attrs.DIRNAME, dirs))),
            dims=(attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME),
            name=attrs.SPECNAME,
        ).to_dataset()

        if tab is not None and attrs.WSPDNAME in tab:
            dset[attrs.WSPDNAME] = xr.DataArray(data=tab[attrs.WSPDNAME].values.reshape(-1,1), dims=[attrs.TIMENAME, attrs.SITENAME])
            dset[attrs.WDIRNAME] = xr.DataArray(data=tab[attrs.WDIRNAME].values.reshape(-1,1), dims=[attrs.TIMENAME, attrs.SITENAME])
        if tab is not None and 'dep' in tab:
            dset[attrs.DEPNAME] = xr.DataArray(data=tab['dep'].values.reshape(-1,1), dims=[attrs.TIMENAME, attrs.SITENAME])

        dset[attrs.LATNAME] = xr.DataArray(data=lats, coords={attrs.SITENAME: sites}, dims=[attrs.SITENAME])
        dset[attrs.LONNAME] = xr.DataArray(data=lons, coords={attrs.SITENAME: sites}, dims=[attrs.SITENAME])

    attrs.set_spec_attributes(dset)
    if 'dir' in dset and len(dset.dir)>1:
        dset[attrs.SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

    return dset

def read_octopus(filename):
    raise NotImplementedError('No Octopus read function defined')

def read_json(self,filename):
    raise NotImplementedError('Cannot read CFJSON format')

def read_swans(fileglob, ndays=None, int_freq=True, int_dir=False, dirorder=True, ntimes=None):
    """
    Read multiple swan files into single Dataset
    Input:
        fileglob :: glob pattern specifying multiple files
        ndays :: maximum number of days from each file to keep, choose None to keep whole time periods
        int_freq :: {array, True, False} frequency array for interpolating onto
            - array: 1d array specifying frequencies to interpolate onto
            - True: logarithm array is constructed so fmin=0.0418 Hz, fmax=0.71856 Hz, df=0.1f
            - False: No interpolation performed in the frequency space (keep it from original spectrum)
        int_dir :: {array, True, False} direction array for interpolating onto
            - array: 1d array specifying directions to interpolate onto
            - True: circular array is constructed so dd=10 degrees
            - False: No interpolation performed in the direction space (keep it from original spectrum)
        dirorder :: if True ensures directions are sorted
        ntimes :: int, use it to read only specific number of times. Useful for checking headers only
    Output:
        SpecDataset object with different sites and cycles concatenated along the 'site' and 'time' dimension
        If multiple cycles are provided, 'time' coordinate is replaced by 'cycletime' multi-index coordinate
    Remarks:
    - If more than one cycle is prescribed from fileglob, each cycle must have same number of sites
    - Either all or none of the spectra in fileglob must have tabfile associated to provide wind/depth data
    - Concatenation is done with numpy arrays for efficiency
    """
    swans = sorted(fileglob) if isinstance(fileglob, list) else sorted(glob.glob(fileglob))
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

    cycles    = list()
    dsets     = SortedDict()
    tabs      = SortedDict()
    all_times = list()
    all_sites = SortedDict()
    all_lons  = SortedDict()
    all_lats  = SortedDict()
    deps      = SortedDict()
    wspds     = SortedDict()
    wdirs     = SortedDict()

    for filename in swans:
        swanfile = SwanSpecFile(filename, dirorder=dirorder)
        times = swanfile.times
        lons = list(swanfile.x)
        lats = list(swanfile.y)
        sites = [os.path.splitext(os.path.basename(filename))[0]] if len(lons)==1 else np.arange(len(lons))+1
        freqs = swanfile.freqs
        dirs = swanfile.dirs

        if ntimes is None:
            spec_list = [s for s in swanfile.readall()]
        else:
            spec_list = [swanfile.read() for itime in range(ntimes)]

        # Read tab files for winds / depth
        if swanfile.is_tab:
            try:
                tab = read_tab(swanfile.tabfile).rename(columns={'dep': attrs.DEPNAME})
                if len(swanfile.times) == tab.index.size:
                    if 'X-wsp' in tab and 'Y-wsp' in tab:
                        tab[attrs.WSPDNAME], tab[attrs.WDIRNAME] = uv_to_spddir(tab['X-wsp'], tab['Y-wsp'], coming_from=True)

                else:
                    print "Warning: times in %s and %s not consistent, not appending winds and depth" % (
                        swanfile.filename, swanfile.tabfile)
                    tab = pd.DataFrame()
                tab = tab[list(set(tab.columns).intersection((attrs.DEPNAME, attrs.WSPDNAME, attrs.WDIRNAME)))]
            except Exception as exc:
                print "Cannot parse depth and winds from %s:\n%s" % (swanfile.tabfile, exc)
        else:
            tab = pd.DataFrame()

        # Shrinking times
        if ndays is not None:
            tend = times[0] + datetime.timedelta(days=ndays)
            if tend > times[-1]:
                raise IOError('Times in %s does not extend for %0.2f days' % (filename, ndays))
            iend = times.index(min(times, key=lambda d: abs(d - tend)))
            times = times[0:iend+1]
            spec_list = spec_list[0:iend+1]
            tab = tab.loc[times[0]:tend] if tab is not None else tab

        spec_list = flatten_list(spec_list, [])

        # Interpolate spectra
        if int_freq is not None or int_dir is not None:
            spec_list = [interp_spec(spec, freqs, dirs, int_freq, int_dir) for spec in spec_list]
            freqs = int_freq if int_freq is not None else freqs
            dirs = int_dir if int_dir is not None else dirs

        # Appending
        try:
            arr = np.array(spec_list).reshape(len(times), len(sites), len(freqs), len(dirs))
            cycle = times[0]
            if cycle not in dsets:
                dsets[cycle] = [arr]
                tabs[cycle] = [tab]
                all_sites[cycle] = sites
                all_lons[cycle] = lons
                all_lats[cycle] = lats
                all_times.append(times)
                nsites = 1
            else:
                dsets[cycle].append(arr)
                tabs[cycle].append(tab)
                all_sites[cycle].extend(sites)
                all_lons[cycle].extend(lons)
                all_lats[cycle].extend(lats)
                nsites += 1
        except:
            if len(spec_list) != arr.shape[0]:
                raise IOError('Time length in %s (%i) does not match previous files (%i), cannot concatenate',
                    (filename, len(spec_list), arr.shape[0]))
            else:
                raise
        swanfile.close()

    cycles = dsets.keys()

    # Ensuring sites are consistent across cycles
    sites = all_sites[cycle]
    lons = all_lons[cycle]
    lats = all_lats[cycle]
    for site, lon, lat in zip(all_sites.values(), all_lons.values(), all_lats.values()):
        if (site != sites) or (lon != lons) or (lat != lats):
            raise IOError('Inconsistent sites across sites in glob pattern provided')

    # Ensuring consistent tabs
    cols = set([frozenset(tabs[cycle][n].columns) for cycle in cycles for n in range(len(tabs[cycle]))])
    if len(cols) > 1:
        raise IOError('Inconsistent tab files, ensure either all or none of the spectra have associated tabfiles and columns are consistent')

    # Concat sites
    for cycle in cycles:
        dsets[cycle] = np.concatenate(dsets[cycle], axis=1)
        deps[cycle] = np.vstack([tab[attrs.DEPNAME].values for tab in tabs[cycle]]).T if attrs.DEPNAME in tabs[cycle][0] else None
        wspds[cycle] = np.vstack([tab[attrs.WSPDNAME].values for tab in tabs[cycle]]).T if attrs.WSPDNAME in tabs[cycle][0] else None
        wdirs[cycle] = np.vstack([tab[attrs.WDIRNAME].values for tab in tabs[cycle]]).T if attrs.WDIRNAME in tabs[cycle][0] else None

    time_sizes = [dsets[cycle].shape[0] for cycle in cycles]

    # Concat cycles
    if len(dsets) > 1:
        dsets = np.concatenate(dsets.values(), axis=0)
        deps = np.concatenate(deps.values(), axis=0) if attrs.DEPNAME in tabs[cycle][0] else None
        wspds = np.concatenate(wspds.values(), axis=0) if attrs.WSPDNAME in tabs[cycle][0] else None
        wdirs = np.concatenate(wdirs.values(), axis=0) if attrs.WDIRNAME in tabs[cycle][0] else None
    else:
        dsets = dsets[cycle]
        deps = deps[cycle] if attrs.DEPNAME in tabs[cycle][0] else None
        wspds = wspds[cycle] if attrs.WSPDNAME in tabs[cycle][0] else None
        wdirs = wdirs[cycle] if attrs.WDIRNAME in tabs[cycle][0] else None

    # Creating dataset
    times = flatten_list(all_times, [])
    dsets = xr.DataArray(
        data=dsets,
        coords=OrderedDict(((attrs.TIMENAME, times), (attrs.SITENAME, sites), (attrs.FREQNAME, freqs), (attrs.DIRNAME, dirs))),
        dims=(attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME),
        name=attrs.SPECNAME,
    ).to_dataset()

    dsets[attrs.LATNAME] = xr.DataArray(data=lats, coords={attrs.SITENAME: sites}, dims=[attrs.SITENAME])
    dsets[attrs.LONNAME] = xr.DataArray(data=lons, coords={attrs.SITENAME: sites}, dims=[attrs.SITENAME])

    if wspds is not None:
        dsets[attrs.WSPDNAME] = xr.DataArray(data=wspds, dims=[attrs.TIMENAME, attrs.SITENAME],
                                       coords=OrderedDict(((attrs.TIMENAME, times), (attrs.SITENAME, sites))))
        dsets[attrs.WDIRNAME] = xr.DataArray(data=wdirs, dims=[attrs.TIMENAME, attrs.SITENAME],
                                       coords=OrderedDict(((attrs.TIMENAME, times), (attrs.SITENAME, sites))))
    if deps is not None:
        dsets[attrs.DEPNAME] = xr.DataArray(data=deps, dims=[attrs.TIMENAME, attrs.SITENAME],
                                      coords=OrderedDict(((attrs.TIMENAME, times), (attrs.SITENAME, sites))))

    # Setting multi-index
    if len(cycles) > 1:
        dsets.rename({'time': 'cycletime'}, inplace=True)
        cycletime = zip([item for sublist in [[c]*t for c,t in zip(cycles, time_sizes)] for item in sublist],
                        dsets.cycletime.values)
        dsets['cycletime'] = pd.MultiIndex.from_tuples(cycletime, names=[attrs.CYCLENAME, attrs.TIMENAME])
        dsets['cycletime'].attrs = attrs.ATTRS[attrs.TIMENAME]

    attrs.set_spec_attributes(dsets)
    if 'dir' in dsets and len(dsets.dir)>1:
        dsets[attrs.SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dsets[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

    return dsets

def read_swanow(fileglob):
    """
    Read SWAN nowcast from fileglob, keep overlapping dates from most recent files
    Inefficient workaround. This should ideally be handled within read_swans by manipulating multi-indexes
    """
    swans = sorted(fileglob) if isinstance(fileglob, list) else sorted(glob.glob(fileglob))
    ds = xr.Dataset()
    for swan in swans:
        ds = read_swan(swan).combine_first(ds)
    return ds

if __name__ == '__main__':

    import datetime
    import matplotlib.pyplot as plt

    ds = read_swan('/source/pyspectra/tests/manus.spec')
    # fileglob = '/mnt/data/work/Hindcast/jogchum/veja/model/swn20161101_??z/*.spec'
    # ds = read_swanow(fileglob)

    # fileglob = '/source/pyspectra/tests/swan/hot/aklislr.20170412_00z.hot-???'
    # fileglob = '/source/pyspectra/tests/swan/hot/aklishr.20170412_12z.hot-???'
    # ds = read_hotswan(fileglob)
    # plt.figure()
    # ds.spec.hs().plot(cmap='jet')
    # plt.show()

    # fileglob = '/source/pyspectra/tests/swan/swn*/*.spec'

    # t0 = datetime.datetime.now()
    # ds = read_swans(fileglob, dirorder=True)
    # print (datetime.datetime.now()-t0).total_seconds()

    # fileglob = '/source/pyspectra/tests/swan/swn20170407_12z/aucki.spec'
    # ds = read_swans(fileglob, dirorder=True)

    # fileglob = '/source/pyspectra/tests/swan/swn20170407_12z/*.spec'
    # ds = read_swans(fileglob, dirorder=True)

