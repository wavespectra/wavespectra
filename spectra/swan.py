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

from scipy.interpolate import griddata

from attributes import *
from misc import to_nautical, D2R

_ = np.newaxis

class Error(Exception):
    pass

class SwanSpecFile(object):
    def __init__(self, filename, freqs=None, dirs=None, x=None, y=None, time=False,
                 id='Swan Spectrum', dirorder=False, append=False, tabfile=None):
        """
        Read spectra in SWAN ASCII format
        """
        self.times = False
        self.filename = filename
        self.tabfile = tabfile or os.path.splitext(self.filename)[0]+'.tab'
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
                self.f = fopen(filename, 'w')
                self.writeHeader(time, id)
                self.fmt = len(self.dirs) * '%4d '
            else:
                self.f = fopen(filename,'r+' if append else 'r')
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
                self.f.readline()
                self.f.readline()

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
            self.buf = self.f.readline()
        if self.buf.find(keyword) >= 0:
            if numspec:
                line = self.f.readline()
                n = int(re.findall(r'\b(\d+)\b', line)[0])
                self.buf = [self.f.readline() for i in range(0,n)]
            rtn = self.buf
            self.buf = None
        else:
            rtn = False
        return rtn

    def read(self):
        if not self.f:
            return None
        if isinstance(self.times, list):
            line = self.f.readline()
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
                    fac = float(self.f.readline())
                    for i,f in enumerate(self.freqs):
                        line = self.f.readline()
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
            line = self.f.readline()
            if not line:
                return -1
            elif line[:15] == tstr:
                self.f.seek(-len(line), 1)
                return i/nf
            i += 1

    def readall(self):
        while True:
            sset = self.read()
            if sset:
                yield sset
            else:
                break

    def writeHeader(self,time=False,str1='',str2=''):
        strout='SWAN   1\n$   '+str1+'\n$   '+str2+'\n'
        if (time):strout+='TIME\n1\n'
        np=len(self.x)
        strout+='LONLAT\n'+str(np)+'\n'
        for i,loc in enumerate(self.x):
            strout+='%f %f\n' % (loc,self.y[i])
        strout += 'AFREQ\n%d\n' % (len(self.freqs))
        for freq in self.freqs:strout+='%f\n' % (freq)

        strout+='NDIR\n%d\n' % (len(self.dirs))
        for dir in self.dirs:strout+='%f\n' % (dir)

        strout+='QUANT\n1\nVaDens\nm2/Hz/degr\n-99\tException value\n'
        self.f.write(strout)

    def writeSpectra(self,specarray):
        for S in specarray:
            fac = S.max()/9998.
            if fac==np.nan:
                strout='NODATA\n'
            elif fac<=0:
                strout='ZERO\n'
            else:
                strout='FACTOR\n'+str(fac)+'\n'
                for row in S:
                    strout+=(self.fmt % tuple(row/fac)) + '\n'
            self.f.write(strout)

    def readSpectrum(self):
        if self.S.any():
            fac = self.S.max()/9998
            if fac<0:return 'NODATA\n'
            strout='FACTOR\n'+str(fac)+'\n'
            for row in self.S:
                strout+=(self.fmt % tuple(row/fac)) + '\n'
            return strout
        else:
            return 'NODATA\n'

    def close(self):
        if self.f:self.f.close()
        self.f=False

def interp_spec(inspec, infreq, indir, outfreq, outdir, method='linear'):
    """
    Interpolate onto new spectral basis
    Input:
        inspec :: 2D numpy array, input spectrum S(infreq,indir) to be interpolated
        infreq :: 1D numpy array, frequencies of input spectrum
        indir :: 1D numpy array, directions of input spectrum
        outfreq :: 1D numpy array, frequencies of output interpolated spectrum
        outdir :: 1D numpy array, directions of output interpolated spectrum
        method :: {'linear', 'nearest', 'cubic'}, method of interpolation to use with griddata
    Output:
        outspec :: 2D numpy array, interpolated ouput spectrum S(outfreq,outdir)
    """
    if (np.array_equal(infreq, outfreq)) & (np.array_equal(indir, outdir)):
        outspec = copy.deepcopy(inspec)
    elif np.array_equal(indir, outdir):
        outspec = np.zeros((len(outfreq), len(outdir)))
        for idir in range(len(indir)):
            outspec[:,idir] = np.interp(outfreq, infreq, inspec[:,idir], left=0., right=0.)
    else:
        dirs = D2R * (270-outdir[_,:])
        dirs2 = D2R * (270-indir[_,:])
        cosmat = np.dot(outfreq[:,_], np.cos(dirs))
        sinmat = np.dot(outfreq[:,_], np.sin(dirs))
        cosmat2 = np.dot(infreq[:,_], np.cos(dirs2))
        sinmat2 = np.dot(infreq[:,_], np.sin(dirs2))
        outspec = griddata((cosmat2.flat, sinmat2.flat), inspec.flat, (cosmat,sinmat), method, 0.)
    return outspec

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
    df.index.name = TIMENAME
    df.index = df.index.shift(toff, freq='1H')
    for col1, col2 in zip(df.columns[-1:0:-1], df.columns[-2::-1]):
        df = df.rename(columns={col2: col1})
    return df.ix[:, 0:-1]


if __name__ == '__main__':


    filename = '/wave/indo/swan_cfsr_west/2012/act.tab'

    t0 = datetime.datetime.now()
    ds1 = read_tab(filename)
    print (datetime.datetime.now()-t0).total_seconds()
