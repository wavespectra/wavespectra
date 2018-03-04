"""Read customised MetOcean Solutions WW3 spectra files."""
import xarray as xr
import numpy as np

from spectra.specdataset import SpecDataset
from spectra.attributes import attrs, set_spec_attributes

def read_ww3_msl(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in MSL netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documentation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'site']
    Returns:
    - dset :: SpecDataset instance
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
