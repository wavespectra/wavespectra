"""Read SWAN spectra files.

Functions:
    read_swan: Read Spectra from SWAN ASCII file
    read_swans: Read multiple swan files into single Dataset
    read_hotswan: Read multiple swan hotfiles into single gridded Dataset
    read_swanow: Read SWAN nowcast from fileglob, keep overlapping dates from most recent files
    read_tab: Read swan table file

Classes:
    SwanSpecFile: methods for reading and writing swan ascii files

"""
import os
import re
import glob
import gzip
import datetime
import pandas as pd
import xarray as xr
import numpy as np
from collections import OrderedDict
from sortedcontainers import SortedDict

from spectra.specdataset import SpecDataset
from spectra.attributes import attrs, set_spec_attributes
from spectra.misc import uv_to_spddir, interp_spec, flatten_list, to_nautical

def read_swan(filename, dirorder=True, as_site=None):
    """Read Spectra from SWAN ASCII file.

    Args:
        dirorder (bool): If True reorder spectra so that directions are sorted
        as_site (bool): If True locations are defined by 1D site dimension
        
    Returns:
        dset (SpecDataset): spectra dataset object read from file

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

    set_spec_attributes(dset)
    if 'dir' in dset and len(dset.dir)>1:
        dset[attrs.SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

    return dset

def read_swans(fileglob, ndays=None, int_freq=True, int_dir=False, dirorder=True, ntimes=None):
    """Read multiple swan files into single Dataset.

    Args:
        fileglob (str, list): glob pattern specifying files to read
        ndays (float): number of days to keep from each file, choose None to keep entire period
        int_freq (ndarray, bool): frequency array for interpolating onto
            - ndarray: 1d array specifying frequencies to interpolate onto
            - True: logarithm array is constructed such that fmin=0.0418 Hz, fmax=0.71856 Hz, df=0.1f
            - False: No interpolation performed in frequency space
        int_dir (ndarray, bool): direction array for interpolating onto
            - ndarray: 1d array specifying directions to interpolate onto
            - True: circular array is constructed such that dd=10 degrees
            - False: No interpolation performed in direction space
        dirorder (bool): if True ensures directions are sorted
        ntimes (int): use it to read only specific number of times, useful for checking headers only
        
    Returns:
        dset (SpecDataset): spectra dataset object read from file with different sites and cycles
                            concatenated along the 'site' and 'time' dimensions

    Note:
        If multiple cycles are provided, 'time' coordinate is replaced by 'cycletime' multi-index coordinate
        If more than one cycle is prescribed from fileglob, each cycle must have same number of sites
        Either all or none of the spectra in fileglob must have tabfile associated to provide wind/depth data
        Concatenation is done with numpy arrays for efficiency

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
        dsets.rename({attrs.TIMENAME: 'cycletime'}, inplace=True)
        cycletime = zip([item for sublist in [[c]*t for c,t in zip(cycles, time_sizes)] for item in sublist],
                        dsets.cycletime.values)
        dsets['cycletime'] = pd.MultiIndex.from_tuples(cycletime, names=[attrs.CYCLENAME, attrs.TIMENAME])
        dsets['cycletime'].attrs = attrs.ATTRS[attrs.TIMENAME]

    set_spec_attributes(dsets)
    if 'dir' in dsets and len(dsets.dir)>1:
        dsets[attrs.SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dsets[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

    return dsets

def read_hotswan(fileglob, dirorder=True):
    """Read multiple swan hotfiles into single gridded Dataset.

    Args:
        fileglob (str, list): glob pattern specifying hotfiles to read and merge
        dirorder (bool): if True ensures directions are sorted

    Returns:
        dset (SpecDataset): spectra dataset object with different grid parts merged

    Note:
        SWAN hotfiles from mpi runs are split by the number of cores over the largest dim of
        (lat, lon) with overlapping rows or columns that are computed in only one of the split
        hotfiles. Here overlappings are merged so that those with higher values are kept
        which assumes non-computed overlapping rows or columns are filled with zeros.

    """
    hotfiles = sorted(fileglob) if isinstance(fileglob, list) else sorted(glob.glob(fileglob))
    assert hotfiles, 'No SWAN file identified with fileglob %s' % (fileglob)

    dsets = [read_swan(hotfiles[0])]
    for hotfile in hotfiles[1:]:
        dset = read_swan(hotfile)
        # Ensure we keep non-zeros in overlapping rows or columns
        overlap = {attrs.LONNAME: set(dsets[-1].lon.values).intersection(dset.lon.values),
                   attrs.LATNAME: set(dsets[-1].lat.values).intersection(dset.lat.values)}
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
    if attrs.DIRNAME in dset and len(dset.dir)>1:
        dset[attrs.SPECNAME].attrs.update({'_units': 'm^{2}.s.degree^{-1}', '_variable_name': 'VaDens'})
    else:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s', '_units': 'm^{2}.s', '_variable_name': 'VaDens'})

    return dset

def read_swanow(fileglob):
    """Read SWAN nowcast from fileglob, keep overlapping dates from most recent files.

    Inefficient workaround. This should ideally be handled within read_swans by manipulating multi-indexes

    """
    swans = sorted(fileglob) if isinstance(fileglob, list) else sorted(glob.glob(fileglob))
    ds = xr.Dataset()
    for swan in swans:
        ds = read_swan(swan).combine_first(ds)
    return ds

def read_tab(filename, toff=0):
    """Read swan table file.

    Args:
        filename (str): name of SWAN tab file to read
        toff (float): timezone offset in hours
    
    Returns:
        Pandas DataFrame object

    """
    dateparse = lambda x: datetime.datetime.strptime(x, '%Y%m%d.%H%M%S')
    df = pd.read_csv(filename,
                     delim_whitespace=True,
                     skiprows=[0,1,2,3,5,6],
                     parse_dates=[0],
                     date_parser=dateparse,
                     index_col=0,
                     )
    df.index.name = attrs.TIMENAME
    df.index = df.index.shift(toff, freq='1H')
    for col1, col2 in zip(df.columns[-1:0:-1], df.columns[-2::-1]):
        df = df.rename(columns={col2: col1})
    return df.ix[:, 0:-1]

class SwanSpecFile(object):
    """Read spectra in SWAN ASCII format."""

    def __init__(self, filename, freqs=None, dirs=None, x=None, y=None, time=False,
                 id='Swan Spectrum', dirorder=False, append=False, tabfile=None):
        self.times = False
        self.filename = filename
        self.tabfile = tabfile or os.path.splitext(self.filename.replace('.gz',''))[0]+'.tab'
        self.is_tab = False
        self.buf = None

        extention = os.path.splitext(self.filename)[-1]
        if extention == '.gz':
            fopen = gzip.open
        else:
            fopen = open
        if freqs is not None: # Writable file
            self.freqs = np.array(freqs)
            self.dirs = np.array(dirs)
            self.x = np.array(x)
            self.y = np.array(y)
            if time:
                self.times = []
            self.fid = fopen(filename, 'w')
            self.write_header(time, id)
            self.fmt = len(self.dirs) * '{:5.0f}'
        else:
            self.fid = fopen(filename,'r+' if append else 'r')
            header = self._read_header('SWAN')
            while True:
                if not self._read_header('$'):
                    break
            if self._read_header('TIME'):
                self._read_header('1')
                self.times = []
            self.x = []
            self.y = []
            for ip in self._read_header('LONLAT', True):
                xy = map(float,ip.split())
                self.x.append(xy[0])
                self.y.append(xy[1])
            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.afreq = self._read_header('AFREQ', True)
            self.rfreq = self._read_header('RFREQ', True)
            self.ndir = self._read_header('NDIR', True)
            self.cdir = self._read_header('CDIR', True)
            self.freqs = np.array(map(float, self.afreq)) if self.afreq else np.array(map(float, self.rfreq))
            if self.ndir:
                self.dirs = np.array(map(float, self.ndir))
            else:
                self.dirs = to_nautical(np.array(map(float, self.cdir)))
            self._read_header('QUANT',True)
            self.fid.readline()
            self.excval = int(float(self.fid.readline().split()[0]))

        if dirorder:
            self.dirmap = list(np.argsort(self.dirs % 360.))
            self.dirs = self.dirs[self.dirmap] % 360.
        else:
            self.dirmap = False
        lons = np.unique(self.x)
        lats = np.unique(self.y)
        self.is_grid = (len(lons)*len(lats) == len(self.x))
        self.is_tab = (os.path.isfile(self.tabfile)) & (len(lons)*len(lats) == 1)

    def _read_header(self, keyword, numspec=False):
        if not self.buf:
            self.buf = self.fid.readline()
        if self.buf.find(keyword) >= 0:
            if numspec:
                line = self.fid.readline()
                n = int(re.findall(r'\b(\d+)\b', line)[0])
                self.buf = [self.fid.readline() for i in range(0,n)]
            rtn = self.buf
            self.buf = None
        else:
            rtn = False
        return rtn

    def read(self):
        """Read single timestep from current position in file."""
        if not self.fid:
            return None
        if isinstance(self.times, list):
            line = self.fid.readline()
            if line:
                ttime = datetime.datetime.strptime(line[0:15], '%Y%m%d.%H%M%S')
                self.times.append(ttime)
            else:
                return None
        Sout = []
        for ip,pp in enumerate(self.x):
            Snew = np.nan * np.zeros((len(self.freqs), len(self.dirs)))
            if self._read_header('NODATA'):
                pass
            else:
                if self._read_header('ZERO'):
                    Snew = np.zeros((len(self.freqs), len(self.dirs)))
                elif self._read_header('FACTOR'):
                    fac = float(self.fid.readline())
                    for i,f in enumerate(self.freqs):
                        line = self.fid.readline()
                        lsplit = line.split()
                        try:
                            Snew[i,:] = map(float, lsplit)
                        except:
                            pass
                    Snew *= fac
                    if self.dirmap:
                        Snew = Snew[:,self.dirmap]
            Sout.append(Snew)
        return Sout

    def readall(self):
        """Read the entire file."""
        while True:
            sset = self.read()
            if sset:
                yield sset
            else:
                break

    def write_header(self, time=False, str1='', str2='', timecode=1, excval=-99):
        """Write header to file."""
        # Description
        strout = '{:40}{}\n'.format('SWAN   1', 'Swan standard spectral file')
        strout += '{:4}{}\n'.format('$', str1)
        strout += '{:4}{}\n'.format('$', str2)
        # Time
        if (time):
            strout += '{:40}{}\n'.format('TIME', 'time-dependent data')
            strout += '{:>6d}{:34}{}\n'.format(timecode, '', 'time coding option')
        # Location
        strout += '{:40}{}\n'.format('LONLAT', 'locations in spherical coordinates')
        strout += '{:>6d}{:34}{}\n'.format(len(self.x), '', 'number of locations')
        for x,y in zip(self.x, self.y):
            strout += '{:2}{:<0.6f}{:2}{:<0.6f}\n'.format('', x, '', y)
        # Frequency
        strout += '{:40}{}\n'.format('AFREQ', 'absolute frequencies in Hz')
        strout += '{:6d}{:34}{}\n'.format(len(self.freqs), '', 'number of frequencies')
        for freq in self.freqs:
            strout += '{:>11.5f}\n'.format(freq)
        # Direction
        strout += '{:40}{}\n'.format('NDIR', 'spectral nautical directions in degr')
        strout += '{:6d}{:34}{}\n'.format(len(self.dirs), '', 'number of directions')
        for wdir in self.dirs:
            strout += '{:>11.4f}\n'.format(wdir)
        # Data
        strout += 'QUANT\n{:>6d}{:34}{}\n'.format(1, '', 'number of quantities in table')
        strout += '{:40}{}\n'.format('VaDens', 'variance densities in m2/Hz/degr')
        strout += '{:40}{}\n'.format('m2/Hz/degr', 'unit')
        strout += '{:3}{:<37g}{}\n'.format('', excval, 'exception value')
        # Dumping
        self.fid.write(strout)

    def write_spectra(self, arr, time=None):
        """Write spectra from single timestamp.

        Args:
            arr (3D ndarray): spectra to write S(site, freq, dim)
            time (datetime): timeof spectra to write

        """
        if time is not None:
            self.fid.write('{:40}{}\n'.format(time.strftime('%Y%m%d.%H%M%S'), 'date and time'))
        for spec in arr:
            fac = spec.max() / 9998.
            if np.isnan(fac):
                strout = 'NODATA\n'
            elif fac <= 0:
                strout = 'ZERO\n'
            else:
                strout = 'FACTOR\n{:4}{:0.8E}\n'.format('', fac)
                for row in spec:
                    strout += self.fmt.format(*tuple(row/fac)) + '\n'
            self.fid.write(strout)

    def close(self):
        """Close file handle."""
        if self.fid:
            self.fid.close()
        self.fid = False

if __name__ == '__main__':
    pass
    # ds = read_swan('/source/pyspectra/tests/manus.spec')
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

