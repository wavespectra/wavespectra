"""Read Native WW3 spectra files."""
import xarray as xr
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes

D2R = np.pi / 180

def read_ww3(filename_or_fileglob, chunks={}):
    """Read Spectra from WAVEWATCHIII native netCDF format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).

    Returns:
        - dset (SpecDataset): spectra dataset object read from ww3 file.

    Note:
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    _units = dset.efth.attrs.get('units', '')
    dset.rename({'frequency': attrs.FREQNAME, 'direction': attrs.DIRNAME,
        'station': attrs.SITENAME, 'efth': attrs.SPECNAME, 'longitude': attrs.LONNAME,
        'latitude': attrs.LATNAME, 'wnddir': attrs.WDIRNAME, 'wnd': attrs.WSPDNAME},
        inplace=True)
    if attrs.TIMENAME in dset[attrs.LONNAME].dims:
        dset[attrs.LONNAME] = dset[attrs.LONNAME].isel(drop=True, **{attrs.TIMENAME: 0})
        dset[attrs.LATNAME] = dset[attrs.LATNAME].isel(drop=True, **{attrs.TIMENAME: 0})
    dset[attrs.SPECNAME] *= D2R
    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update({'_units': _units, '_variable_name': attrs.SPECNAME})
    if attrs.DIRNAME not in dset or len(dset.dir)==1:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s'})
    dset[attrs.DIRNAME] = (dset[attrs.DIRNAME]+180) % 360
    return dset