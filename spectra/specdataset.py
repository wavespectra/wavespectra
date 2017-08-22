"""
Extra functions to attach to main SpecArray class
"""
import re
import xarray as xr
import numpy as np
from cfjson.xrdataset import CFJSONinterface
from specarray import SpecArray
from attributes import *
from swan import SwanSpecFile
from misc import to_datetime

OCT_MISSING=-99999

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

    def to_swan(self, filename, append=False, id='Created by pyspectra'):
        """
        Write spectra in SWAN ASCII format
        Input:
            - filename :: str, name for output SWAN ASCII file
            - append :: if True append to existing filename
            - id :: str, used for header in output file
        Remark:
            - Only datasets with lat/lon coordinates are currently supported
            - Only 2D spectra E(f,d) are currently supported
            - Extra dimensions (other than time, site, lon, lat, freq, dim) are not yet supported
        """
        darray = self.dset['efth'].copy(deep=False)
        is_time = 'time' in darray.dims

        # If grid reshape into site, if neither define fake site dimension
        if set(('lon','lat')).issubset(darray.dims):
            darray = darray.stack(site=('lat','lon'))
        elif 'site' not in darray.dims:
            darray = darray.expand_dims('site')

        # Intantiate swan object
        try:
            x = darray.lon.values
            y = darray.lat.values
        except NotImplementedError('lon/lat not found in dset, cannot dump SWAN file without locations'):
            raise
        sfile = SwanSpecFile(filename, freqs=darray.freq, dirs=darray.dir,
                             time=is_time, x=x, y=y, append=append, id=id)

        # Dump each timestep
        if is_time:
            for t in darray.time:
                sfile.writeSpectra(darray.sel(time=t).transpose('site','freq','dir').values,
                                   time=to_datetime(t.values))
        else:
            sfile.writeSpectra(darray.sel(time=t).transpose('site','freq','dir').values)
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
            wd = self.wdir.values if hasattr(self,'wdir') else OCT_MISSING*np.ones(len(self.time))
            ws = self.wspd.values if hasattr(self,'wspd') else OCT_MISSING*np.ones(len(self.time))
            dpt = self.dpt.values if hasattr(self,'dpt') else OCT_MISSING*np.ones(len(self.time))
            for i, t in enumerate(self.time):
                if i == 0:
                    f.write('nfreqs,%d\nndir,%d\nnrecs,%d\nLatitude,%7.4f\nLongitude,%7.4f\nDepth,%f\n\n' %
                            (len(self.freq),
                             len(self.dir),
                             len(self.time),
                             self.lat[0].values,
                             self.lon[0].values,
                             dpt[0]) )
                    sdirs = np.mod(self.dir.values, 360.)
                    idirs = np.argsort(sdirs)
                    ddir = abs(sdirs[1] - sdirs[0])
                    dfreq = np.hstack((0, self.freq[1:].values - self.freq[:-1].values,0))
                    dfreq = 0.5 * (dfreq[:-1] + dfreq[1:])
                lp = site_id + to_datetime(t).strftime('_%Y%m%d_%Hz')
                s_sea = sea[i].spec
                s_sw = swell[i].spec
                s = self.isel(time=i).efth.squeeze().spec
                f.write('CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n')
                f.write('%s,\'%s,%s,%d,%.2f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.4f,%.2f,%.1f,%.5f,%.5f,%.4f,%d,%d,%d\n' %
                        (to_datetime(t).strftime('%Y%m'), to_datetime(t).strftime('%d%H%M'), lp, wd[i], ws[i],
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

if __name__ == '__main__':
    from specdataset import SpecDataset
    from readspec import read_swanow
    fileglob = '/mnt/data/work/Hindcast/jogchum/veja/model/swn20161101_??z/*.spec'
    ds = read_swanow(fileglob)
    ds.spec.to_swan('/home/rafael/tmp/test2.spec')

    from spectra.readspec import read_swan
    ds = read_swan('/source/pyspectra/tests/antf0.20170208_06z.hot-001')
    ds.spec.to_swan('/home/rafael/tmp/test.spec')