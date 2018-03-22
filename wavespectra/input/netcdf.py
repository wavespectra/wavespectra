"""Read generic netCDF spectra files."""
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes

def read_netcdf(filename_or_fileglob,
                chunks={},
                freqname=attrs.FREQNAME,
                dirname=attrs.DIRNAME,
                sitename=attrs.SITENAME,
                specname=attrs.SPECNAME,
                lonname=attrs.LONNAME,
                latname=attrs.LATNAME,
                timename=attrs.TIMENAME):
    """Read Spectra from generic netCDF format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).
        - <coord>name :: coordinate name in netcdf, used for standarising
          dataset.

    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file

    Note:
        - Assumes frequency in :math:`Hz`, direction in :math:`degree` and
          spectral energy in :math:`m^{2}degree^{-1}{s}`.
        - If file is large to fit in memory, consider specifying chunks for
          'time' and/or 'station' dims.

    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    _units = dset[specname].attrs.get('units','')
    _variable_name = specname
    coord_map = {freqname: attrs.FREQNAME,
                 dirname: attrs.DIRNAME,
                 lonname: attrs.LONNAME,
                 latname: attrs.LATNAME,
                 sitename: attrs.SITENAME,
                 specname: attrs.SPECNAME,
                 timename: attrs.TIMENAME}
    dset.rename({k:v for k,v in coord_map.items() if k in dset}, inplace=True)
    dset[attrs.SPECNAME].attrs.update({'_units': _units, '_variable_name': _variable_name})
    if attrs.DIRNAME not in dset or len(dset.dir)==1:
        dset[attrs.SPECNAME].attrs.update({'units': 'm^{2}.s'})
    set_spec_attributes(dset)
    return dset