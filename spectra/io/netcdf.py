#Generic netcdf in/output

import xarray as xr
from spectra.io.attributes import *

def read_netcdf(filename_or_fileglob,
                chunks={},
                freqname=FREQNAME,
                dirname=DIRNAME,
                sitename=SITENAME,
                specname=SPECNAME,
                lonname=LONNAME,
                latname=LATNAME):
    """
    Read Spectra off generic netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'station']
    Returns:
    - dset :: SpecDataset instance
    -----
    Note that this current assumes frequency as Hz, direction as degrees and spectral energy as m^2s
    """
    from spectra import SpecDataset

    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    if sitename in dset.dims: #Sites based
        dset.rename({freqname: FREQNAME, dirname: DIRNAME, sitename: SITENAME, specname: SPECNAME}, inplace=True)
    else: #Gridded
        dset.rename({freqname: FREQNAME, dirname: DIRNAME, lonname: LONNAME, latname: LATNAME, specname: SPECNAME}, inplace=True)
    set_spec_attributes(dset)
    return SpecDataset(dset)

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