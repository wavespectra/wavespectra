"""
Testing new spectra object on WW3 native spectra files
"""
import xarray as xr
from pyspectra import NewSpecArray

filename = './snative20141201T00Z_spec.nc'

def read_spec_ww3_native(filename):
    """
    Read Spectra off WW3 in native netCDF format
    Returns:
    - spec_array :: DataArray object with spectra methods in the spec accessor
    """
    with xr.open_dataset(filename) as dset:
        # The following loads into memory, we may want to review how we access the dataarray
        # Maybe we return the dataset anhdinstead of the dataarray
        spec_array = dset['efth'].rename({'frequency': 'freq', 'direction': 'dir'}).load()
    return spec_array


da = read_spec_ww3_native(filename)

for method in ['hs', 'tp', 'tm01', 'tm02', 'dm', 'dp', 'dpm', 'dspr', 'swe', 'sw']:
    print 'Cheking out method %s' % (method)
    print getattr(da.spec, method)().isel(time=0)