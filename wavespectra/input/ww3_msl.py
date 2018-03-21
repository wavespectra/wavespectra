"""Read customised MetOcean Solutions WW3 spectra files."""
import xarray as xr
import numpy as np

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes

def read_ww3_msl(filename_or_fileglob, chunks={}):
    """Read Spectra from WAVEWATCHIII MetOcean Solutions netCDF format.

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
          'time' and/or 'site' dims

    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    _units = dset.specden.attrs.get('units','')
    dset.rename({'freq': attrs.FREQNAME, 'dir': attrs.DIRNAME, 'wsp': attrs.WSPDNAME}, inplace=True)
    dset[attrs.SPECNAME] = (dset['specden'].astype('float32')+127.) * dset['factor']
    dset = dset.drop(['specden', 'factor', 'df'])
    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update({'_units': _units, '_variable_name': 'specden'})
    if attrs.DIRNAME not in dset or len(dset.dir)==1:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s'})
    return dset