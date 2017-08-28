"""
Extra functions to attach to main SpecArray class
"""
import re
import xarray as xr
import numpy as np
from tqdm import tqdm
from cfjson.xrdataset import CFJSONinterface
from specarray import SpecArray
from attributes import *
from swan import SwanSpecFile
from misc import to_datetime

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
        darray = self.dset['efth'].copy(deep=True)
        is_time = 'time' in darray.dims

        # If grid reshape into site, if neither define fake site dimension
        if set(('lon','lat')).issubset(darray.dims):
            darray = darray.stack(site=('lat','lon'))
        elif 'site' not in darray.dims:
            darray = darray.expand_dims('site')

        # Intantiate swan object
        try:
            x = self.dset.lon.values
            y = self.dset.lat.values
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

    def to_octopus(self, filename, site_id='spec', fcut=0.125, missing_val=-99999):
        """
        Save spectra in Octopus format
        """
        if ('site' in self.dims) and len(self.site)>1:
            raise NotImplementedError('No Octopus export defined for multiple sites')
        elif ('lon' in self.dims) and (len(self.lon)>1 or len(self.lat)>1):
            raise NotImplementedError('No Octopus export defined for grids')
        
        # Update dataset with parameters
        stats = ['hs', 'tm01', 'dm']
        dset = xr.merge([self.dset,
                         self.spec.stats(stats+['dpm','dspr']),
                         self.spec.stats(stats, names=[s+'_swell' for s in stats], fmax=fcut),
                         self.spec.stats(stats, names=[s+'_sea' for s in stats], fmin=fcut),
                         self.spec.momf(mom=1).sum(dim='dir').rename('momf1'),
                         self.spec.momf(mom=2).sum(dim='dir').rename('momf2'),
                         self.spec.momd(mom=0)[0].rename('momd'),
                         ]).sortby('dir')
        if WDIRNAME not in dset:
            dset[WDIRNAME] = 0 * dset['hs'] + missing_val
        if WSPDNAME not in dset:
            dset[WSPDNAME] = 0 * dset['hs'] + missing_val
        if DEPNAME not in dset:
            dset[DEPNAME] = 0 * dset['hs'] + missing_val

        fmt = ','.join(len(self.freq)*['{:6.5f}']) + ','
        fmt2 = '\n'.join(len(self.dir) * ['{:d},' + fmt])
        fmt3 = '{:d},' + fmt# + '{:6.5f},'
        # fmt2 = '\n'.join(len(self.dir) * ['{:d},' + fmt + '{:6.5f},'])
        dt = (to_datetime(dset.time[1]) - to_datetime(dset.time[0])).total_seconds() / 3600

        # Start site loop here
        lat = float(self.lat[0])
        lon = float(self.lon[0])
        
        with open(filename, 'w') as f:

            # General header
            f.write('Forecast valid for {:%d-%b-%Y %H:%M:%S}\n'.format(to_datetime(self.time[0])))
            f.write('nfreqs,{:d}\n'.format(len(self.freq)))
            f.write('ndir,{:d}\n'.format(len(self.dir)))
            f.write('nrecs,{:d}\n'.format(len(self.time)))
            f.write('Latitude,{:0.6f}\n'.format(lat))
            f.write('Longitude,{:0.6f}\n'.format(lon))
            f.write('Depth,{:0.2f}\n\n'.format(float(dset[DEPNAME].isel(time=0))))

            # Frequency resolution
            sdirs = np.mod(self.dir.values, 360.)
            idirs = np.argsort(sdirs)
            # ddir = abs(sdirs[1] - sdirs[0])
            dd = dset.efth.spec.dd
            dfreq = np.hstack((0, dset['efth'].spec.df, 0))
            dfreq = 0.5 * (dfreq[:-1] + dfreq[1:])

            # Dump each timestep
            for i, t in enumerate(tqdm(self.time)):
                ds = dset.isel(time=i)
                # Timestamp header
                lp = '{}_{:%Y%m%d_%Hz}'.format(site_id, to_datetime(t))
                f.write('CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n')
                # Header and parameters
                f.write("{:%Y%m,'%d%H%M},{},{:d},{:.2f},{:.4f},{:.2f},{:.1f},{:.4f},{:.2f},{:.1f},{:.4f},{:.2f},{:.1f},{:.5f},{:.5f},{:.4f},{:d},{:d},{:d}\n".format(
                            to_datetime(t), lp, int(ds[WDIRNAME]), float(ds[WSPDNAME]),
                            0.25*float(ds['hs'])**2, float(ds['tm01']), float(ds['dm']),
                            0.25*float(ds['hs_sea'])**2, float(ds['tm01_sea']), float(ds['dm_sea']),
                            0.25*float(ds['hs_swell'])**2, float(ds['tm01_swell']), float(ds['dm_swell']),
                            float(ds['momf1']), float(ds['momf2']), float(ds['hs']), int(ds['dpm']), int(ds['dspr']), int(i*dt)))
                # Frequencies
                f.write(('freq,'+fmt+'anspec\n').format(*ds.freq.values))
                # Spectra
                energy = ds.spec.to_energy().squeeze().T
                specdump = ''
                for direc in self.dir[0:1]:
                    row = np.hstack((direc, energy.sel(dir=direc).values))
                    specdump += '{:d},'.format(int(direc))
                    specdump += fmt.format(*row)
                    specdump += '{:6.5f},\n'.format(row.sum())
                f.write(specdump)
                # import ipdb; ipdb.set_trace()
                f.write(('fSpec,' + fmt + '\n').format(*(self.efth.spec.dfarr.values * df.momd.squeeze().values))) 
                
                # np.savetxt(f, x, fmt=fmt2)
                    # spec = 
                # for idir in idirs:
                #     f.write('%d,' % (np.round(sdirs[idir])))
                #     row = dset.spec.dd * dfreq * s._obj.isel(dir=idir)
                #     f.write(fmt.format(*row.values))
                #     f.write('%6.5f,\n' % row.sum())
                # f.write(('fSpec,'+fmt+'\n').format(*dfreq*s.momd(0)[0].values))
                # f.write(('den,'+fmt+'\n\n').format(*s.momd(0)[0].values))

if __name__ == '__main__':
    from specdataset import SpecDataset
    from readspec import read_swan
    fileglob = '/source/pyspectra/tests/manus.spec'
    ds = read_swan(fileglob)
    ds.spec.to_octopus('/Users/rafaguedes/tmp/test.oct')

    # from spectra.readspec import read_swan
    # ds = read_swan('/source/pyspectra/tests/antf0.20170208_06z.hot-001')
    # ds.spec.to_swan('/home/rafael/tmp/test.spec')