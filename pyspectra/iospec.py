"""
Access functions to read spectra from different file formats
"""
import xarray as xr
from pyspectra.spectra import NewSpecArray

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


if __name__ == '__main__':
    filename = './tests/snative20141201T00Z_spec.nc'
    da = read_spec_ww3_native(filename)