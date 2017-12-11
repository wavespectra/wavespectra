"""
Extra functions to attach to main SpecArray class
"""
import re
import sys
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

    def _to_dump(self, supported_dims=['time','site','lat','lon','freq','dir']):
        """
        Ensure dimensions are suitable for dumping in some ascii formats
        Input:
            - supported_dims :: list of dimensions that are supported by the dumping method
        Returns:
            - Dataset object with site dimension and with no grid dimensions
        Remark:
            - grid is converted to site dimension which can be iterated over
            - site is defined if not in dataset and not a grid
            - spectral coordinates are checked to ensure they are supported for dumping
        """
        dset = self.dset.copy(deep=True)

        unsupported_dims = set(dset.dims) - set(supported_dims)
        if unsupported_dims:
            raise NotImplementedError('Dimensions {} are not supported by {} method'.format(
                unsupported_dims, sys._getframe().f_back.f_code.co_name))

        # If grid reshape into site, if neither define fake site dimension
        if set(('lon','lat')).issubset(dset.dims):
            dset = dset.stack(site=('lat','lon'))
        elif 'site' not in dset.dims:
            dset = dset.expand_dims('site')

        return dset

    def to_swan(self, filename, append=False, id='Created by pyspectra', unique_times=False):
        """
        Write spectra in SWAN ASCII format
        Input:
            - filename :: str, name for output SWAN ASCII file
            - append :: if True append to existing filename
            - id :: str, used for header in output file
            - unique_times :: if True, only last time is taken from duplicate indices
        Remark:
            - Only datasets with lat/lon coordinates are currently supported
            - Only 2D spectra E(f,d) are currently supported
            - Extra dimensions (other than time, site, lon, lat, freq, dim) are not yet supported
        """
        # If grid reshape into site, otherwise ensure there is site dim to iterate over
        dset = self._to_dump()
        darray = dset['efth']
        is_time = 'time' in darray.dims

        # Instantiate swan object
        try:
            x = dset.lon.values
            y = dset.lat.values
        except NotImplementedError('lon/lat not found in dset, cannot dump SWAN file without locations'):
            raise
        sfile = SwanSpecFile(filename, freqs=darray.freq, dirs=darray.dir,
                             time=is_time, x=x, y=y, append=append, id=id)

        # Dump each timestep
        if is_time:
            for t in darray.time:
                darrout = darray.sel(time=t)
                if darrout.time.size == 1:
                    sfile.writeSpectra(darrout.transpose('site','freq','dir').values,
                                       time=to_datetime(t.values))
                elif unique_times:
                    sfile.writeSpectra(darrout.isel(time=-1).transpose('site','freq','dir').values,
                                       time=to_datetime(t.values))
                else:
                    for it,tt in enumerate(darrout.time):
                            sfile.writeSpectra(darrout.isel(time=it).transpose('site','freq','dir').values,
                                               time=to_datetime(t.values))
        else:
            sfile.writeSpectra(darray.transpose('site','freq','dir').values)
        sfile.close()
    

    def to_ww3(self, filename):
        raise NotImplementedError('Cannot write to native WW3 format')

    def to_ww3_msl(filename):
        raise NotImplementedError('Cannot write to native WW3 format')

    def to_octopus(self, filename, site_id='spec', fcut=0.125, missing_val=-99999):
        """
        Save spectra in Octopus format
        Input:
            - filename :: str, name for output OCTOPUS file
            - site_id :: str, used to construct LPoint header
            - fcut :: float, frequency for splitting spectra to calculate some parameters
            - missing_value :: int, missing value in output file
        Remark:
            - dataset needs to have lon/lat/time coordinates
            - 1D spectra are not supported
            - dataset with multiple locations is dumped at same file with one location header per site
        """
        assert 'time' in self.dims, "Octopus output requires time dimension"

        # If grid reshape into site, otherwise ensure there is site dim to iterate over
        dset = self._to_dump()

        fmt = ','.join(len(self.freq)*['{:6.5f}']) + ','
        if len(dset.time) > 1:
            dt = (to_datetime(dset.time[1]) - to_datetime(dset.time[0])).total_seconds() / 3600.
        else:
            dt = 0.

        # Open output file
        with open(filename, 'w') as f:

            # Looping over each site
            for isite in range(len(dset.site)):
                dsite = dset.isel(site=[isite])

                # Site coordinates
                try:
                    lat = float(dsite.lat)
                    lon = float(dsite.lon)
                except NotImplementedError('lon/lat not found in dset, cannot dump OCTOPUS file without locations'):
                    raise
                
                # Update dataset with parameters
                stats = ['hs', 'tm01', 'dm']
                dsite = xr.merge([dsite,
                                dsite.spec.stats(stats+['dpm','dspr']),
                                dsite.spec.stats(stats, names=[s+'_swell' for s in stats], fmax=fcut),
                                dsite.spec.stats(stats, names=[s+'_sea' for s in stats], fmin=fcut),
                                dsite.spec.momf(mom=1).sum(dim='dir').rename('momf1'),
                                dsite.spec.momf(mom=2).sum(dim='dir').rename('momf2'),
                                dsite.spec.momd(mom=0)[0].rename('momd'),
                                dsite.spec.to_energy(),
                                (dsite.efth.spec.dfarr * dsite.spec.momd(mom=0)[0]).rename('fSpec'),
                                ]).sortby('dir').fillna(missing_val)

                if WDIRNAME not in dsite:
                    dsite[WDIRNAME] = 0 * dsite['hs'] + missing_val
                if WSPDNAME not in dsite:
                    dsite[WSPDNAME] = 0 * dsite['hs'] + missing_val
                if DEPNAME not in dsite:
                    dsite[DEPNAME] = 0 * dsite['hs'] + missing_val
                
                # General header
                f.write('Forecast valid for {:%d-%b-%Y %H:%M:%S}\n'.format(to_datetime(dsite.time[0])))
                f.write('nfreqs,{:d}\n'.format(len(dsite.freq)))
                f.write('ndir,{:d}\n'.format(len(dsite.dir)))
                f.write('nrecs,{:d}\n'.format(len(dsite.time)))
                f.write('Latitude,{:0.6f}\n'.format(lat))
                f.write('Longitude,{:0.6f}\n'.format(lon))
                f.write('Depth,{:0.2f}\n\n'.format(float(dsite[DEPNAME].isel(time=0))))

                # Dump each timestep
                for i, t in enumerate(dsite.time):
                    ds = dsite.isel(time=i)

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
                                        
                    # Spectra
                    energy = ds['energy'].squeeze().T.values
                    specdump = ''
                    for idir,direc in enumerate(self.dir):
                        row = energy[idir]
                        specdump += '{:d},'.format(int(direc))
                        specdump += fmt.format(*row)
                        specdump += '{:6.5f},\n'.format(row.sum())
                    f.write(('freq,'+fmt+'anspec\n').format(*ds.freq.values))
                    f.write(specdump)
                    f.write(('fSpec,' + fmt + '\n').format(*ds['fSpec'].squeeze().values))
                    f.write(('den,' + fmt + '\n\n').format(*ds['momd'].squeeze().values))

if __name__ == '__main__':
    # from specdataset import SpecDataset
    from readspec import read_swan
    fileglob = '/source/pyspectra/tests/manus.spec'
    ds = read_swan(fileglob)
    # ds.spec.to_octopus('/Users/rafaguedes/tmp/test.oct')
    # ds.spec.to_octopus('/home/rafael/tmp/test.oct')
    ds.spec.to_swan('/home/rafael/tmp/test.swn')

    # from spectra.readspec import read_swan
    # ds = read_swan('/source/pyspectra/tests/antf0.20170208_06z.hot-001')
    # ds.spec.to_swan('/home/rafael/tmp/test.spec')
    # ds.spec.to_octopus('/home/rafael/tmp/test.oct')