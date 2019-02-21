"""Read Native WWM netCDF spectra files."""
import xarray as xr
import numpy as np
import dask.array as da

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.misc import uv_to_spddir

R2D = 180 / np.pi

def read_wwm(filename_or_fileglob, chunks={}, convert_wind_vectors=True):
    """Read Spectra from SWAN native netCDF format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - convert_wind_vectors (bool): choose it to convert wind vectors into
          speed / direction data arrays.

    Returns:
        - dset (SpecDataset): spectra dataset object read from ww3 file.

    Note:
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    _units = dset.AC.attrs.get('units', '')
    dset = dset.rename({
        'nfreq': attrs.FREQNAME,
        'ndir': attrs.DIRNAME,
        'nbstation': attrs.SITENAME,
        'AC': attrs.SPECNAME,
        'lon': attrs.LONNAME,
        'lat': attrs.LATNAME,
        'DEP': attrs.DEPNAME,
        'ocean_time': attrs.TIMENAME
    })
    # Calculating wind speeds and directions
    if convert_wind_vectors and 'Uwind' in dset and 'Vwind' in dset:
        dset[attrs.WSPDNAME], dset[attrs.WDIRNAME] = uv_to_spddir(
            dset['Uwind'], dset['Vwind'], coming_from=True)
    # Setting standard names and storing original file attributes
    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update({'_units': _units, '_variable_name': attrs.SPECNAME})
    # Assigning spectral coordinates
    dset[attrs.FREQNAME] = dset.SPSIG / (2 * np.pi) # convert rad to Hz
    dset[attrs.DIRNAME] = dset.SPDIR
    # converting Action to Energy density and adjust density to Hz
    dset[attrs.SPECNAME] = dset[attrs.SPECNAME] * dset.SPSIG * (2 * np.pi)
    # Converting from radians
    dset[attrs.DIRNAME] *= R2D
    dset[attrs.SPECNAME] /= R2D
    # Returns only selected variables, transposed
    to_drop = [dvar for dvar in dset.data_vars if dvar not in [attrs.SPECNAME,
        attrs.WSPDNAME, attrs.WDIRNAME, attrs.DEPNAME, attrs.LONNAME, attrs.LATNAME]]
    dims = [d for d in ['time','site','freq','dir'] if d in dset.efth.dims]
    return dset.drop(to_drop).transpose(*dims)

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import xarray as xr

    FILES_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../tests/sample_files'
    )
    dset = read_wwm(os.path.join(FILES_DIR, 'wwmfile.nc'))

    ds = xr.open_dataset('/source/wavespectra/tests/sample_files/wwmfile.nc')
    hs_wwm = ds.HS
    tp_wwm = ds.TPP
    hs_wavespectra = dset.spec.spec.hs()
    tp_wavespectra = dset.spec.spec.tp()

    plt.figure()
    hs_wavespectra.isel(site=0).plot(label='wavespectra')
    hs_wwm.isel(nbstation=0).plot(label='wwm')
    plt.title('Hs')
    plt.legend()

    plt.figure()
    tp_wavespectra.isel(site=0).plot(label='wavespectra')
    tp_wwm.isel(nbstation=0).plot(label='wwm')
    plt.title('Tp')
    plt.legend()

    s = dset.isel(site=0, time=5).rename({'freq': 'period'})
    s.period.values = 1. / s.period.values
    plt.figure()
    s.efth.plot()
    print 'Tp from file: {}'.format(ds.isel(nbstation=0, ocean_time=5).TPP.values)

    plt.show()
