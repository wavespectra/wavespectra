# pyspectra
Python library for wave spectra

## Main contents:
- [spectra.specarray.SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31): object based on [xarray.DataArray](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html) to manipulate spectra and calculate spectral statistics
- [spectra.specdataset.SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16): wrapper around [SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31) with methods for saving spectra in different formats
- [spectra.readspec](https://github.com/metocean/pyspectra/blob/master/spectra/readspec.py): access functions to read spectra from different file formats into [SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16) objects

## Code structure
The two main classes [spectra.specarray.SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31) and [spectra.specdataset.SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16) are defined as xarray accessors as described [here](http://xarray.pydata.org/en/stable/internals.html?highlight=accessor). The accessors are registered on [xarray.DataArray](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html) and [xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html) respectively as a new namespace `spec`.

To use methods in the accessor classes simply import the classes into your code and they will be available to your xarray.Dataset or xarray.DataArray instances through the `spec` attribute, *e.g.*
```python
import datetime
import numpy as np
import xarray as xr

from spectra.specarray import SpecArray
from spectra.specdataset import SpecDataset

coords = {'time': [datetime.datetime(2017,01,n+1) for n in range(2)],
          'freq': [0.05,0.1],
          'dir': np.arange(0,360,120)}
efth = xr.DataArray(data=np.random.rand(2,2,3), coords=coords, dims=('time','freq', 'dir'))

In [1]: efth
Out[1]:
<xarray.DataArray (time: 2, freq: 2, dir: 3)>
array([[[ 0.100607,  0.328229,  0.332708],
        [ 0.532   ,  0.665938,  0.177731]],

       [[ 0.469371,  0.002963,  0.627179],
        [ 0.004523,  0.682717,  0.09766 ]]])
Coordinates:
  * freq     (freq) float64 0.05 0.1
  * dir      (dir) int64 0 120 240
  * time     (time) datetime64[ns] 2017-01-01 2017-01-02

In [2]: efth.spec
Out[2]:
<SpecArray (time: 2, freq: 2, dir: 3)>
array([[[ 0.100607,  0.328229,  0.332708],
        [ 0.532   ,  0.665938,  0.177731]],

       [[ 0.469371,  0.002963,  0.627179],
        [ 0.004523,  0.682717,  0.09766 ]]])
Coordinates:
  * freq     (freq) float64 0.05 0.1
  * dir      (dir) int64 0 120 240
  * time     (time) datetime64[ns] 2017-01-01 2017-01-02
  
In [3]: efth.spec.hs()
Out[3]:
<xarray.DataArray 'hs' (time: 2)>
array([ 10.128485,   9.510618])
Coordinates:
  * time     (time) datetime64[ns] 2017-01-01 2017-01-02
Attributes:
    standard_name: sea_surface_wave_significant_height
    units: m
```
## Data requirements
[SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31) methods require the following attributes in [xarray.DataArray](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html):
- wave frequency coordinate in `Hz` named as `freq` (required)
- wave direction coordinate in `degree` (coming from) named as `dir` (optional for 1D, required for 2D spectra)
- wave spectra data in `m2/Hz/degree` (2D) or `m2/Hz` (1D)

[SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16) methods require the following attributes in [xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html):
- spectra DataArray named as `efth` complying to the specifications above

SpecDataset provides a wrapper around the methods in SpecArray accessor. For instance, both of these would produce the same result:
```python
dset = efth.to_dataset(name='efth')
tm01_specarr = dset.efth.spec.tm01()
tm01_specdset = dset.spec.tm01()

In [3]: tm01_specarr
Out[3]: 
<xarray.DataArray 'tm01' (time: 2)>
array([ 13.341924,  13.129259])
Coordinates:
  * time     (time) datetime64[ns] 2017-01-01 2017-01-02
Attributes:
    standard_name: sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment
    units: s
```

## Examples

### Define and plot spectra history from example [SWAN spectra](http://swanmodel.sourceforge.net/online_doc/swanuse/node50.html) file:
```python
from spectra import read_swan

dset = read_swan('/source/pyspectra/tests/manus.spec')
spec_hist = dset.isel(lat=0, lon=0).sel(freq=slice(0.05,0.2)).spec.oned().T
spec_hist.plot.contourf(levels=10)
```
