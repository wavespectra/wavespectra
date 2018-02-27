"""
Auxiliary class to parse spectra from SWAN ASCII format
"""
import os
import re
import copy
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import gzip

import spectra.attributes as attrs
from spectra.misc import to_nautical

class Error(Exception):
    pass

class SwanSpecFile(object):
    def __init__(self, filename, freqs=None, dirs=None, x=None, y=None, time=False,
                 id='Swan Spectrum', dirorder=False, append=False, tabfile=None):
        """
        Read spectra in SWAN ASCII format
        TODO: Exception value is parsed but is not applied when reading spectra. Make sure this is ok
        """
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
        try:
            if freqs is not None:#Writable file
                self.freqs = np.array(freqs)
                self.dirs = np.array(dirs)
                self.x = np.array(x)
                self.y = np.array(y)
                if time:
                    self.times = []
                self.fid = fopen(filename, 'w')
                self.writeHeader(time, id)
                self.fmt = len(self.dirs) * '{:5.0f}'
            else:
                self.fid = fopen(filename,'r+' if append else 'r')
                header = self._readhdr('SWAN')
                while True:
                    if not self._readhdr('$'):
                        break
                if self._readhdr('TIME'):
                    self._readhdr('1')
                    self.times = []
                self.x = []
                self.y = []
                for ip in self._readhdr('LONLAT', True):
                    xy = map(float,ip.split())
                    self.x.append(xy[0])
                    self.y.append(xy[1])
                self.x = np.array(self.x)
                self.y = np.array(self.y)

                self.afreq = self._readhdr('AFREQ', True)
                self.rfreq = self._readhdr('RFREQ', True)
                self.ndir = self._readhdr('NDIR', True)
                self.cdir = self._readhdr('CDIR', True)
                self.freqs = np.array(map(float, self.afreq)) if self.afreq else np.array(map(float, self.rfreq))
                if self.ndir:
                    self.dirs = np.array(map(float, self.ndir))
                else:
                    self.dirs = to_nautical(np.array(map(float, self.cdir)))

                self._readhdr('QUANT',True)
                self.fid.readline()
                self.excval = int(float(self.fid.readline().split()[0]))

        except Error as e:
            raise 'File error with %s [%s]' % (filename, e)
        if dirorder:
            self.dirmap = list(np.argsort(self.dirs % 360.))
            self.dirs = self.dirs[self.dirmap] % 360.
        else:
            self.dirmap = False
        lons = np.unique(self.x)
        lats = np.unique(self.y)
        self.is_grid = (len(lons)*len(lats) == len(self.x))
        self.is_tab = (os.path.isfile(self.tabfile)) & (len(lons)*len(lats) == 1)

    def _readhdr(self, keyword, numspec=False):
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
            if self._readhdr('NODATA'):
                pass
            else:
                if self._readhdr('ZERO'):
                    Snew = np.zeros((len(self.freqs), len(self.dirs)))
                elif self._readhdr('FACTOR'):
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

    def scan(self, time):
        nf = len(self.S.freqs) + 1
        tstr = time.strftime('%Y%m%d.%H%M%S')
        i = 0
        while True:
            line = self.fid.readline()
            if not line:
                return -1
            elif line[:15] == tstr:
                self.fid.seek(-len(line), 1)
                return i/nf
            i += 1

    def readall(self):
        while True:
            sset = self.read()
            if sset:
                yield sset
            else:
                break

    def writeHeader(self, time=False, str1='', str2='', timecode=1, excval=-99):
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

    def writeSpectra(self, arr, time=None):
        """
        Dump spectra from single timestamp into SWAN object
            - arr :: 3D numpy array arr(site,freq,dim)
            - time :: datetime object for current timestamp
        """
        if time is not None:
            self.fid.write('{:40}{}\n'.format(time.strftime('%Y%m%d.%H%M%S'), 'date and time'))
        for spec in arr:
            fac = spec.max()/9998.
            if np.isnan(fac):
                strout = 'NODATA\n'
            elif fac <= 0:
                strout = 'ZERO\n'
            else:
                strout = 'FACTOR\n{:4}{:0.8E}\n'.format('', fac)
                for row in spec:
                    strout += self.fmt.format(*tuple(row/fac)) + '\n'
            self.fid.write(strout)

    def readSpectrum(self):
        if self.S.any():
            fac = self.S.max()/9998
            if fac < 0:
                return 'NODATA\n'
            strout = 'FACTOR\n{:4}{:0.8E}\n'.format('', fac)
            for row in self.S:
                strout += self.fmt.format(*tuple(row/fac)) + '\n'
            return strout
        else:
            return 'NODATA\n'

    def close(self):
        if self.fid:
            self.fid.close()
        self.fid = False

def read_tab(filename, toff=0):
    """
    Read swan tab file, return pandas dataframe
    Usage:
        df = read_swan_tab(filename, mask={}, toff=0)
    Input:
        filename :: name of SWAN tab file to read
        toff :: timezone offset
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


if __name__ == '__main__':


    filename = '/wave/indo/swan_cfsr_west/2012/act.tab'

    t0 = datetime.datetime.now()
    ds1 = read_tab(filename)
    print (datetime.datetime.now()-t0).total_seconds()
