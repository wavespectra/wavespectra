"""
Access functions to read spectra from different file formats
"""
from collections import OrderedDict
import numpy as np
import xarray as xr
from pyspectra.spectra import NewSpecArray

# TODO: Function to read and merge multiple SWAN hotfiles

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
    - dset :: Dataset with spec accessor class attached to spectra DataArray
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    dset.rename({'frequency': FREQNAME, 'direction': DIRNAME, 'station': SITENAME, 'efth': SPECNAME}, inplace=True)
    dset[SPECNAME].attrs = SPECATTRS
    dset[SPECNAME].values = np.radians(dset[SPECNAME].values)
    return dset

def read_spec_ww3_msl(filename_or_fileglob, chunks={}):
    """
    Read Spectra off WW3 in MSL netCDF format
    - filename_or_fileglob :: either filename or fileglob specifying multiple files
    - chunks :: Chunking dictionary specifying chunk sizes for dimensions for dataset into dask arrays.
                By default dataset is loaded using single chunk for all arrays (see xr.open_mfdataset documtation)
                Typical dimensions in native WW3 netCDF considering chunking are: ['time', 'site']
    Returns:
    - dset :: Dataset with spec accessor class attached to spectra DataArray
    """
    dset = xr.open_mfdataset(filename_or_fileglob, chunks=chunks)
    dset.rename({'freq': FREQNAME, 'dir': DIRNAME}, inplace=True)#, 'SITE': SITENAME})
    dset[SPECNAME] = (dset['specden'].astype('float32')+127.) * dset['factor']
    dset[SPECNAME].attrs = SPECATTRS
    dset = dset.drop(['specden','factor', 'df'])
    return dset

def read_spec_swan(filename, dirorder=True):
    """
    Read Spectra off SWAN ASCII file
    - dirorder :: If True reorder spectra read from file so that directions are sorted
    Returns:
    - dset :: Dataset with spec accessor class attached to spectra DataArray
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
    lons = np.unique(flat_lons)
    lats = np.unique(flat_lats)
    if len(lons)*len(lats) == len(flat_lons):
        # Looks like gridded data, grid DataArray accordingly
        arr = np.array([s.S for s in spec_list]).reshape(len(times), len(lons), len(lats), len(freqs), len(dirs))
        dset = xr.DataArray(
            data=np.swapaxes(arr, 1, 2),
            coords=OrderedDict(((TIMENAME, times), (LATNAME, lats), (LONNAME, lons), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, LATNAME, LONNAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            attrs=SPECATTRS,
            ).to_dataset()
    else:
        # Keep it with sites dimension
        dset = xr.DataArray(
            data=np.array([s.S for s in spec_list]).reshape(len(times), len(sites), len(freqs), len(dirs)),
            coords=OrderedDict(((TIMENAME, times), (SITENAME, sites), (FREQNAME, freqs), (DIRNAME, dirs))),
            dims=(TIMENAME, SITENAME, FREQNAME, DIRNAME),
            name=SPECNAME,
            attrs=SPECATTRS,
        ).to_dataset()
        dset[LATNAME] = xr.DataArray(data=flat_lats, coords={SITENAME: sites}, dims=[SITENAME])
        dset[LONNAME] = xr.DataArray(data=flat_lons, coords={SITENAME: sites}, dims=[SITENAME])

    return dset


if __name__ == '__main__':
    # Reading native WW3 spectra, do not subchunk it
    filename = './tests/snative20141201T00Z_spec.nc'
    dset_native = read_spec_ww3_native(filename)

    # Reading MSL WW3 spectra, use small chunk size for site
    filename = '/wave/global/ww3_0.5_tc/s20000101_00z.nc'
    dset_msl = read_spec_ww3_msl(filename, chunks={'site': 10})

    # Reading SWAN hotfile into a regular grid
    filename = './tests/antf0.20170208_06z.hot-001'
    dset_swanhot = read_spec_swan(filename)
    dset_swanhot.isel(time=0)['efth'].spec.hs().plot()