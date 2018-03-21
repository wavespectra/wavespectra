"""SWAN ASCII output plugin."""
from wavespectra.core.attributes import attrs
from wavespectra.core.swan import SwanSpecFile
from wavespectra.core.misc import to_datetime

def to_swan(self, filename, append=False, id='Created by wavespectra', unique_times=False):
    """Write spectra in SWAN ASCII format.

    Args:
        - filename (str): str, name for output SWAN ASCII file.
        - append (bool): if True append to existing filename.
        - id (str): used for header in output file.
        - unique_times (bool): if True, only last time is taken from
          duplicate indices.

    Note:
        - Only datasets with lat/lon coordinates are currently supported.
        - Extra dimensions other than time, site, lon, lat, freq, dim not yet
          supported.
        - Only 2D spectra E(f,d) are currently supported.

    """
    # If grid reshape into site, otherwise ensure there is site dim to iterate over
    dset = self._check_and_stack_dims()
    
    darray = dset[attrs.SPECNAME]
    is_time = attrs.TIMENAME in darray.dims

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
            darrout = darray.sel(time=t, method='nearest')
            if darrout.time.size == 1:
                sfile.write_spectra(darrout.transpose(attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME).values,
                                    time=to_datetime(t.values))
            elif unique_times:
                sfile.write_spectra(darrout.isel(time=-1).transpose(attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME).values,
                                    time=to_datetime(t.values))
            else:
                for it,tt in enumerate(darrout.time):
                    sfile.write_spectra(darrout.isel(time=it).transpose(attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME).values,
                                        time=to_datetime(t.values))
    else:
        sfile.write_spectra(darray.transpose(attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME).values)
    sfile.close()