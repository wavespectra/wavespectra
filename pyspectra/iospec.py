"""
Access functions to read spectra from different file formats
"""
from collections import OrderedDict
import numpy as np
import xarray as xr
from pyspectra.spectra import NewSpecArray

# TODO: Ensure lon and lat are attached to dataarray for cases where they are not coordinates

SPECNAME = 'efth'
TIMENAME = 'time'
SITENAME = 'site'
LATNAME = 'lat'
LONNAME = 'lon'
FREQNAME = 'freq'
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
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'station']
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
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'site']
    Returns:
    - spec_array :: DataArray object with spectra methods in the spec accessor
    - dset :: Dataset handle
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    spec_array = (dset['specden'].astype('float32')+127.) * dset['factor']
    spec_array = spec_array.rename({'freq': FREQNAME, 'dir': DIRNAME})#, 'SITE': SITENAME})
    spec_array.attrs = SPECATTRS
    return spec_array.rename(SPECNAME), dset

def read_spec_swan(filename, dirorder=True, grid=False):
    """
    Read Spectra off SWAN ASCII file
    """
    def flatten(l, a):
        """
        Flatten list of lists
        """
        for i in l:
            if isinstance(i, list):
                flatten(i, a)
            else:
                a.append(i)
        return a

    # Read up spectra arrays and coordinates
    from legacy import SwanSpecFile2 # Avoid pymo dependency in top-level module
    spectra = SwanSpecFile2(filename, dirorder=True)
    spec_list = flatten([s for s in spectra.readall()], [])

    # Sort out missing data which is read as an one-dimensional array
    times = spectra.times
    sites = spectra.locations
    freqs = spec_list[0].freqs
    dirs = spec_list[0].dirs
    flat_lons = [site.x for site in spectra.locations]
    flat_lats = [site.y for site in spectra.locations]

    # Assign DataArray
    if grid:
        lons = np.unique(flat_lons)
        lats = np.unique(flat_lats)
        assert len(lons)*len(lats) == len(flat_lons), "Grid cannot be constructed from unique coordinates in locations attribute"
        arr = np.array([s.S for s in spec_list]).reshape(len(times), len(lons), len(lats), len(freqs), len(dirs))
        spec_array = xr.DataArray(
            data=np.swapaxes(arr, 1, 2),
            coords=OrderedDict(((TIMENAME, times), (LATNAME, lats), (LONNAME, lons), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, LATNAME, LONNAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            attrs=SPECATTRS,
            )
    else:
        spec_array = xr.DataArray(
            data=np.array([s.S for s in spec_list]).reshape(len(times), len(sites), len(freqs), len(dirs)),
            coords=OrderedDict(((TIMENAME, times), (SITENAME, sites), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, SITENAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            attrs=SPECATTRS,
        )

    return spec_array


if __name__ == '__main__':
    # Reading native WW3 spectra, do not subchunk it
    filename = './tests/snative20141201T00Z_spec.nc'
    da_msl, dset_msl = read_spec_ww3_native(filename)

    # Reading MSL WW3 spectra, use small chunk size for site
    filename = '/wave/global/ww3_0.5_tc/s20000101_00z.nc'
    da_native, dset_native = read_spec_ww3_native(filename, chunks={site: 10})

    # Reading SWAN hotfile into a regular grid
    filename = './tests/antf0.20170208_06z.hot-001'
    da_swanhot = read_spec_swan(filename, grid=True)
    da_swanhot.isel(time=0).spec.hs().plot()