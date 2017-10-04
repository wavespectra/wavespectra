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
from tqdm import tqdm
from scipy.io import loadmat
from dbfread import DBF

from swan import SwanSpecFile, read_tab
from specdataset import SpecDataset
from attributes import *
from misc import uv_to_spddir, to_datetime, interp_spec, flatten_list, to_nautical, dnum_to_datetime

def read_triaxys_direc(filename_or_fileglob, toff=0):
    """
    Read 2D wave spectra file from TRIAXYS buoy
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - toff :: time offset in hours to account for time zone differences
    Returns:
    - dset :: SpecDataset instance
    Remark: frequencies and directions from first file are used as reference
            to interpolate spectra from other files in case they differ
    """
    is_triaxys = False
    is_directional = False
    time = None
    nf = None
    df = None
    f0 = None
    ddir = None
    times = list()
    spec_list = list()

    filenames = sorted(filename_or_fileglob) if isinstance(filename_or_fileglob, list) else sorted(glob.glob(filename_or_fileglob))
    assert filenames, "No file found to read"

    for ifile, filename in enumerate(tqdm(filenames)):
        with open(filename, 'r') as f:
            # Parsing header
            for line in f:
                if 'TRIAXYS BUOY DATA REPORT' in line or 'TRIAXYS BUOY REPORT' in line:
                    is_triaxys = True
                if 'DIRECTIONAL SPECTRUM' in line:
                    is_directional = True
                if 'DATE' in line:
                    time = parser.parse(line.split('=')[1].split('(')[0].strip()) - datetime.timedelta(hours=toff)
                if 'NUMBER OF FREQUENCIES' in line:
                    nf = int(line.split('=')[1])
                if 'INITIAL FREQUENCY' in line:
                    f0 = float(line.split('=')[1])
                if 'FREQUENCY SPACING' in line:
                    df = float(line.split('=')[1])
                if 'DIRECTION SPACING' in line:
                    ddir = float(line.split('=')[1])
                if 'ROWS' in line:
                    break
            if not is_triaxys:
                raise IOError('Not a TRIAXYS Spectra file.')
            if not is_directional:
                raise IOError('Not a directional Spectra file.')
            if not time:
                raise IOError('Cannot parse time')
            if ddir:
                dirs = list(pd.np.arange(0., 360.+ddir, ddir))
            else:
                raise IOError('Not enough info to parse directions')
            if None not in [nf, f0, df]:
                freqs = list(pd.np.arange(f0, df*nf, df))
            else:
                raise IOError('Not enough info to parse frequencies')

            # Parsing data
            spec = pd.np.zeros((len(freqs), len(dirs)))
            for i in range(nf):
                spec[i,:] = map(float, f.next().split())

            # Ensure same spectral basis
            if ifile == 0:
                interp_freq = copy.deepcopy(freqs)
                interp_dir = copy.deepcopy(dirs)
            spec_list.append(interp_spec(inspec=spec,
                                         infreq=freqs,
                                         indir=dirs,
                                         outfreq=interp_freq,
                                         outdir=interp_dir))
            times.append(time)

    # Construct dataset
    dset = xr.DataArray(
            data=spec_list,
            coords=OrderedDict(((TIMENAME, times), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            ).to_dataset()
    set_spec_attributes(dset)
    return dset

def read_triaxys_dbf(filename_or_fileglob):
    """
    Read 1D spectra off dbf TRIAXYS files of type:
    | DATE | TIME | STATUS | SpectraCnt | FirstFreq | FreqSpacin | Field1 | Field2 | ... | FieldN |

    Returns:
    - dset :: SpecDataset instance (1D, no direction dimension included)
    """
    dbfs = sorted(filename_or_fileglob) if isinstance(filename_or_fileglob, list) else sorted(glob.glob(filename_or_fileglob))

    spec_list = list()
    times = list()
    for ifile,filename in enumerate(tqdm(dbfs)):
        dframe = pd.DataFrame(iter(DBF(filename)))
        dframe.set_index(pd.to_datetime(dframe['DATE']) + pd.to_timedelta(dframe['TIME']), inplace=True)
        dframe.drop(['TIME','DATE'], axis=1, inplace=True)
        if ifile == 0:
            interp_freq = np.arange(dframe['FirstFreq'].min(),
                                    dframe['FirstFreq'].max() + dframe['SpectraCnt'].max()*dframe['FreqSpacin'].min(),
                                    dframe['FreqSpacin'].min())
        for index, row in dframe.iterrows():
            vals = [row[col] for col in row.keys() if 'Field' in col]
            if len(vals) != int(row['SpectraCnt']):
                raise IOError('Spectrum array does not match SpectraCnt for %s' % (index))
            freq = np.arange(row['FirstFreq'],
                             row['FirstFreq'] + row['SpectraCnt']*row['FreqSpacin'],
                             row['FreqSpacin'])[0:int(row['SpectraCnt'])]
            spec_list.append(interp_spec(inspec=vals, infreq=freq, indir=None, outfreq=interp_freq))
        times.extend(dframe.index.tolist())

    dset = xr.DataArray(
            data=spec_list,
            coords=OrderedDict(((TIMENAME, times), (FREQNAME, interp_freq))),
            dims=(TIMENAME, FREQNAME),
            name=SPECNAME,
            ).to_dataset()
    dset[SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})
    return dset

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
    coord_map = {freqname: FREQNAME,
                 dirname: DIRNAME,
                 lonname: LONNAME,
                 latname: LATNAME,
                 sitename: SITENAME,
                 specname: SPECNAME}
    dset.rename({k:v for k,v in coord_map.items() if k in dset}, inplace=True)
    dset[SPECNAME].attrs.update({'_units': _units, '_variable_name': _variable_name})
    if 'dir' not in dset or len(dset.dir)==1:
        dset[SPECNAME].attrs.update({'units': 'm^{2}.s'})
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
    dset.rename({'frequency': FREQNAME, 'direction': DIRNAME,
        'station': SITENAME, 'efth': SPECNAME, 'longitude': LONNAME,
        'latitude': LATNAME, 'wnddir': WDIRNAME, 'wnd': WSPDNAME},
        inplace=True)
    if TIMENAME in dset[LONNAME].dims:
        dset[LONNAME] = dset[LONNAME].isel(drop=True, **{TIMENAME: 0})
        dset[LATNAME] = dset[LATNAME].isel(drop=True, **{TIMENAME: 0})
    dset[SPECNAME].values = np.radians(dset[SPECNAME].values)
    set_spec_attributes(dset)
    dset[SPECNAME].attrs.update({'_units': _units, '_variable_name': 'efth'})
    if 'dir' not in dset or len(dset.dir)==1:
        dset[SPECNAME].attrs.update({'units': 'm^{2}.s'})
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
    dset.rename({'freq': FREQNAME, 'dir': DIRNAME, 'wsp': WSPDNAME}, inplace=True)#, 'SITE': SITENAME})
    dset[SPECNAME] = (dset['specden'].astype('float32')+127.) * dset['factor']
    dset = dset.drop(['specden','factor', 'df'])
    set_spec_attributes(dset)
    dset[SPECNAME].attrs.update({'_units': _units, '_variable_name': 'specden'})
    if 'dir' not in dset or len(dset.dir)==1:
        dset[SPECNAME].attrs.update({'units': 'm^{2}.s'})
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
    if 'dir' in dset and len(dset.dir)>1:
        dset[SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dset[SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

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
    if 'dir' in dset and len(dset.dir)>1:
        dset[SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dset[SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

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

    for filename in tqdm(swans):
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
                tab = read_tab(swanfile.tabfile).rename(columns={'dep': DEPNAME})
                if len(swanfile.times) == tab.index.size:
                    if 'X-wsp' in tab and 'Y-wsp' in tab:
                        tab[WSPDNAME], tab[WDIRNAME] = uv_to_spddir(tab['X-wsp'], tab['Y-wsp'], coming_from=True)

                else:
                    print "Warning: times in %s and %s not consistent, not appending winds and depth" % (
                        swanfile.filename, swanfile.tabfile)
                    tab = pd.DataFrame()
                tab = tab[list(set(tab.columns).intersection((DEPNAME, WSPDNAME, WDIRNAME)))]
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
        deps[cycle] = np.vstack([tab[DEPNAME].values for tab in tabs[cycle]]).T if DEPNAME in tabs[cycle][0] else None
        wspds[cycle] = np.vstack([tab[WSPDNAME].values for tab in tabs[cycle]]).T if WSPDNAME in tabs[cycle][0] else None
        wdirs[cycle] = np.vstack([tab[WDIRNAME].values for tab in tabs[cycle]]).T if WDIRNAME in tabs[cycle][0] else None

    time_sizes = [dsets[cycle].shape[0] for cycle in cycles]

    # Concat cycles
    if len(dsets) > 1:
        dsets = np.concatenate(dsets.values(), axis=0)
        deps = np.concatenate(deps.values(), axis=0) if DEPNAME in tabs[cycle][0] else None
        wspds = np.concatenate(wspds.values(), axis=0) if WSPDNAME in tabs[cycle][0] else None
        wdirs = np.concatenate(wdirs.values(), axis=0) if WDIRNAME in tabs[cycle][0] else None
    else:
        dsets = dsets[cycle]
        deps = deps[cycle] if DEPNAME in tabs[cycle][0] else None
        wspds = wspds[cycle] if WSPDNAME in tabs[cycle][0] else None
        wdirs = wdirs[cycle] if WDIRNAME in tabs[cycle][0] else None

    # Creating dataset
    times = flatten_list(all_times, [])
    dsets = xr.DataArray(
        data=dsets,
        coords=OrderedDict(((TIMENAME, times), (SITENAME, sites), (FREQNAME, freqs), (DIRNAME, dirs))),
        dims=(TIMENAME, SITENAME, FREQNAME, DIRNAME),
        name=SPECNAME,
    ).to_dataset()

    dsets[LATNAME] = xr.DataArray(data=lats, coords={SITENAME: sites}, dims=[SITENAME])
    dsets[LONNAME] = xr.DataArray(data=lons, coords={SITENAME: sites}, dims=[SITENAME])

    if wspds is not None:
        dsets[WSPDNAME] = xr.DataArray(data=wspds, dims=[TIMENAME, SITENAME],
                                       coords=OrderedDict(((TIMENAME, times), (SITENAME, sites))))
        dsets[WDIRNAME] = xr.DataArray(data=wdirs, dims=[TIMENAME, SITENAME],
                                       coords=OrderedDict(((TIMENAME, times), (SITENAME, sites))))
    if deps is not None:
        dsets[DEPNAME] = xr.DataArray(data=deps, dims=[TIMENAME, SITENAME],
                                      coords=OrderedDict(((TIMENAME, times), (SITENAME, sites))))

    # Setting multi-index
    if len(cycles) > 1:
        dsets.rename({'time': 'cycletime'}, inplace=True)
        cycletime = zip([item for sublist in [[c]*t for c,t in zip(cycles, time_sizes)] for item in sublist],
                        dsets.cycletime.values)
        dsets['cycletime'] = pd.MultiIndex.from_tuples(cycletime, names=[CYCLENAME, TIMENAME])
        dsets['cycletime'].attrs = ATTRS[TIMENAME]

    set_spec_attributes(dsets)
    if 'dir' in dsets and len(dsets.dir)>1:
        dsets[SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dsets[SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

    return dsets

def read_swanow(fileglob):
    """
    Read SWAN nowcast from fileglob, keep overlapping dates from most recent files
    Inefficient workaround. This should ideally be handled within read_swans by manipulating multi-indexes
    """
    swans = sorted(fileglob) if isinstance(fileglob, list) else sorted(glob.glob(fileglob))
    ds = xr.Dataset()
    for swan in tqdm(swans):
        ds = read_swan(swan).combine_first(ds)
    return ds

def read_diwasp(filename, struct_name='SMout', dnum_name='Dnum', qc=True):
    """
    Read spectra from DIWASP Matlab structure
    Input:
    - struct_name :: name of matlab data struct saved in the matfile
    - dnum_name :: name of matlab datenum time vector saved in the matfile
    - qc :: Bool, if True basic qc is performed before returning spectra
    Returns:
    - dset :: SpecDataset instance
    """
    # Reading marfile, expecting struct with spectra and array with datenums
    mat = loadmat(filename)
    smout = mat[struct_name][0]
    dnum = mat[dnum_name][0]
    # Assigning spectra info
    freqs = set([tuple(s[0].squeeze()) for s in smout if any(s[0])])
    dirs = set([tuple(s[1].squeeze()) for s in smout if any(s[1])])
    spec_list = [s[2] for d,s in zip(dnum,smout) if any(s[2].ravel()) and d>0]
    times = [dnum_to_datetime(d) for d,s in zip(dnum,smout) if any(s[2].ravel()) and d>0]
    xdir = set([float(s[3]) for s in smout if any(s[3])])
    units = set([str(s[4].squeeze()) for s in smout if any(s[3])])
    # Sanity check
    if len(smout) != len(dnum):
        raise Exception('Lenght of times and spectra arrays differ')
    if (len(freqs) == 1) & (len(dirs) == 1) & (len(xdir) == 1) & (len(units) == 1):
        freqs = np.array(list(freqs)).ravel()
        dirs = np.array(list(dirs)).ravel()
        xdir = list(xdir)[0]
        units = list(units)[0]
    else:
        raise Exception('Frequecies or directions not consistent across spectra')
    # Diwasp output may have last direction duplicated
    if dirs[0] % 360 == dirs[-1] % 360:
        dirs = dirs[:-1]
        spec_list = [s[:,:-1] for s in spec_list]
    # Ensure Nautical convention
    if xdir == 90:
        dirs = np.vectorize(to_nautical)(dirs)
    # Define specdset
    dset = xr.DataArray(
        data=spec_list,
        coords=OrderedDict(((TIMENAME, times), (FREQNAME, freqs), (DIRNAME, dirs))),
        dims=(TIMENAME, FREQNAME, DIRNAME),
        name=SPECNAME,
        ).to_dataset().sel(**{DIRNAME: sorted(dirs)})
    dset[SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})
    set_spec_attributes(dset)
    # Basic QC
    dset = dset.where((dset.spec.hs()>0) & (dset.spec.hs()<100))
    return dset

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

