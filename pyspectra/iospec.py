"""
Access functions to read spectra from different file formats
"""
from collections import OrderedDict
import xarray as xr
from pyspectra.spectra import NewSpecArray

SPECNAME = 'efth'
FREQNAME = 'freq'
SITENAME = 'site'
DIRNAME = 'dir'
SPECATTRS = OrderedDict((
    ('standard_name', 'sea_surface_wave_directional_variance_spectral_density'),
    ('units', 'm2s')
    ))

def read_spec_ww3_native(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in native netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
    Returns:
    - spec_array :: DataArray object with spectra methods in the spec accessor
    - dset :: Dataset handle
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    spec_array = dset['efth'].rename({'frequency': FREQNAME, 'direction': DIRNAME, 'station': SITENAME})
    spec_array.attrs = SPECATTRS
    return spec_array.rename(SPECNAME), dset

def read_spec_ww3_msl(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in MSL netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation) 
    Returns:
    - spec_array :: DataArray object with spectra methods in the spec accessor
    - dset :: Dataset handle
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    spec_array = (dset['specden'].astype('float32')+127.) * dset['factor']
    spec_array = spec_array.rename({'freq': FREQNAME, 'dir': DIRNAME})#, 'SITE': SITENAME})
    spec_array.attrs = SPECATTRS
    return spec_array.rename(SPECNAME), dset


if __name__ == '__main__':
    filename = './tests/snative20141201T00Z_spec.nc'
    da = read_spec_ww3_native(filename)