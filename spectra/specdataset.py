"""
Extra functions to attach to main SpecArray class
"""
import re
import xarray as xr
import numpy as np

from specarray import SpecArray
from attributes import *
from swan import SwanSpecFile

@xr.register_dataset_accessor('spec')
class SpecDataset(object):
    """
    Provides a wrapper around the xarray dataset 
    """
    def __init__(self, xarray_dset):
        self.dset = xarray_dset
        
    def __repr__(self):
        return re.sub(r'<.+>', '<%s>'%(self.__class__.__name__), str(self.dset))
    
    def __getattr__(self, fn):
        if fn in dir(SpecArray) and (fn[0] != '_'):
            return getattr(self.dset['efth'].spec, fn)
        else:
            return getattr(self.dset, fn)

    def to_json(self, filename, attributes={}):
        strout = self.dset.cfjson.json_dumps(indent=2, attributes=attributes)
        with open(filename,'w') as f:
            f.write(strout)

    def to_netcdf(self, filename,
                  specname='efth',
                  ncformat='NETCDF4_CLASSIC',
                  compress=True,
                  time_encoding={'units': 'days since 1900-01-01'}):
        """
        Preset parameters before calling xarray's native to_netcdf method
        - specname :: name of spectra variable in dataset
        - ncformat :: netcdf format for output, see options in native to_netcdf method
        - compress :: if True output is compressed, has no effect for NETCDF3
        - time_encoding :: force standard time units in output files
        """
        other = self.copy(deep=True)
        encoding = {}
        if compress:
            for ncvar in other.data_vars:
                encoding.update({ncvar: {'zlib': True}})
        if 'time' in other:
            other.time.encoding.update(time_encoding)
        other.to_netcdf(filename, format=ncformat, encoding=encoding)

    def to_swan(self,filename, id='Swan Spectrum', append=False):
        if 'site' in self.dset.dims:
            xx = self.lon
            yy = self.lat
        else:
            xx = self.lon
            yy = self.lat
        sfile = SwanSpecFile(filename, freqs=self.freq, time=True, dirs=self.dir,
                             x=self.lon, y=self.lat, id=id, append=append)
        for i,ct in enumerate(self.time):
            sfile.f.write(to_datetime(t.values).strftime('%Y%m%d.%H%M%S\n'))
            sfile.writeSpectra(self.dset['efth'].sel(time=t).values.reshape(-1, len(self.freq), len(self.dir)))
        sfile.close()

    def to_ww3(self, filename):
        raise NotImplementedError('Cannot write to native WW3 format')

    def to_ww3_msl(filename):
        raise NotImplementedError('Cannot write to native WW3 format')

    def to_octopus(self, filename, site_id='spec'):
        """
        Save spectra in Octopus format
        """
        if ('site' in self.dims) and len(self.site)>1:
            raise NotImplementedError('No Octopus export defined for multiple sites')
        elif ('lon' in self.dims) and (len(self.lon)>1 or len(self.lat)>1):
            raise NotImplementedError('No Octopus export defined for grids')
        
        with open(filename, 'w') as f:
            f.write('Forecast valid for %s\n' % (to_datetime(self.time[0]).strftime('%d-%b-%Y %H:%M:%S')))
            fmt = ','.join(len(self.freq)*['%6.5f'])+','
            dt = (self.time[1].astype('int') - self.time[0].astype('int')) / 3.6e12 if len(self.time)>1 else 0
            swell = self.split(fmin=0, fmax=0.125)
            sea = self.split(fmin=0.125, fmax=1.)
            for i, t in enumerate(self.time):
                s = self.efth
                if i == 0:
                    f.write('nfreqs,%d\nndir,%d\nnrecs,%d\nLatitude, %7.4f\nLongitude, %7.4f\nDepth,%f\n\n' %
                            (len(self.freq), len(self.dir), len(self.time), self.lat[0], self.lon[0], self.dpt[0]))
                    sdirs = np.mod(self.dir.values, 360.)
                    idirs = np.argsort(sdirs)
                    ddir = abs(sdirs[1] - sdirs[0])
                    dfreq = np.hstack((0, self.freq[1:].values - self.freq[:-1].values,0))
                    dfreq = 0.5 * (dfreq[:-1] + dfreq[1:])
                lp = site_id + to_datetime(t).strftime('_%Y%m%d_%Hz')
                wd = self.wnddir[i]
                ws = self.wnd[i]
                s_sea = sea[i].spec
                s_sw = swell[i].spec
                s = self.isel(time=i).efth.squeeze().spec
                f.write('CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n')
                f.write('%s,\'%s,%s,%d,%.2f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.5f,%.5f,%.4f,%d,%d,%d\n' %
                        (to_datetime(t).strftime('%Y%m'), to_datetime(t).strftime('%d%H%M'), lp, wd, ws,
                         (0.25*s.hs())**2, s.tm01(), s.dm(),
                         (0.25*s_sea.hs())**2, s_sea.tm01(), s_sea.dm(),
                         (0.25*s_sw.hs())**2, s_sw.tm01(), s_sw.dm(),
                         s.momf(1).sum(), s.momf(2).sum(), s.hs(), s.dpm(), s.dspr(), i*dt))
                f.write(('freq,'+fmt+'anspec\n') % tuple(self.freq))
                for idir in idirs:
                    f.write('%d,' % (np.round(sdirs[idir])))
                    row = ddir * dfreq * s._obj.isel(dir=idir)
                    f.write(fmt % tuple(row.values))
                    f.write('%6.5f,\n' % row.sum())
                f.write(('fSpec,'+fmt+'\n') % tuple(dfreq*s.momd(0)[0].values))
                f.write(('den,'+fmt+'\n\n') % tuple(s.momd(0)[0].values))
