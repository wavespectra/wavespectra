"""OCTOPUS output plugin."""
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core.misc import to_datetime

def to_octopus(self, filename, site_id='spec', fcut=0.125, missing_val=-99999):
    """Save spectra in Octopus format.

    Args:
        - filename (str): name for output OCTOPUS file.
        - site_id (str): used to construct LPoint header.
        - fcut (float): frequency for splitting spectra.
        - missing_value (int): missing value in output file.

    Note:
        - dataset needs to have lon/lat/time coordinates.
        - dataset with multiple locations dumped at same file with one location
          header per site.
        - 1D spectra not supported.

    """
    assert attrs.TIMENAME in self.dims, "Octopus output requires time dimension"

    # If grid reshape into site, otherwise ensure there is site dim to iterate over
    dset = self._check_and_stack_dims()

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

            if attrs.WDIRNAME not in dsite:
                dsite[attrs.WDIRNAME] = 0 * dsite['hs'] + missing_val
            if attrs.WSPDNAME not in dsite:
                dsite[attrs.WSPDNAME] = 0 * dsite['hs'] + missing_val
            if attrs.DEPNAME not in dsite:
                dsite[attrs.DEPNAME] = 0 * dsite['hs'] + missing_val
            
            # General header
            f.write('Forecast valid for {:%d-%b-%Y %H:%M:%S}\n'.format(to_datetime(dsite.time[0])))
            f.write('nfreqs,{:d}\n'.format(len(dsite.freq)))
            f.write('ndir,{:d}\n'.format(len(dsite.dir)))
            f.write('nrecs,{:d}\n'.format(len(dsite.time)))
            f.write('Latitude,{:0.6f}\n'.format(lat))
            f.write('Longitude,{:0.6f}\n'.format(lon))
            f.write('Depth,{:0.2f}\n\n'.format(float(dsite[attrs.DEPNAME].isel(time=0))))

            # Dump each timestep
            for i, t in enumerate(dsite.time):
                ds = dsite.isel(time=i)

                # Timestamp header
                lp = '{}_{:%Y%m%d_%Hz}'.format(site_id, to_datetime(t))
                f.write('CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n')

                # Header and parameters
                f.write("{:%Y%m,'%d%H%M},{},{:d},{:.2f},{:.4f},{:.2f},{:.1f},{:.4f},{:.2f},{:.1f},{:.4f},{:.2f},{:.1f},{:.5f},{:.5f},{:.4f},{:d},{:d},{:d}\n".format(
                            to_datetime(t), lp, int(ds[attrs.WDIRNAME]), float(ds[attrs.WSPDNAME]),
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