# pyspectra
Python library for wave spectra

## Main contents:
- [spectra.specarray.SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31): object based on [xarray.DataArray](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html) to manipulate spectra and calculate spectral statistics
- [spectra.specdataset.SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16): wrapper around [SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31) with methods for saving spectra in different formats
- [spectra.readspec](https://github.com/metocean/pyspectra/blob/master/spectra/readspec.py): access functions to read spectra from different file formats into [SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16) objects

## Structure
The two main classes [spectra.specarray.SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31) and [spectra.specdataset.SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16) are defined as xarray accessors as described [here](http://xarray.pydata.org/en/stable/internals.html?highlight=accessor). The accessors are registered on [xarray.DataArray](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html) and [xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html) respectively as a new namespace `spec`.

To use methods in the accessor classes simply import the classes into your code:
```python
from spectra.specarray import SpecArray
from spectra.specdataset import SpecDataset
```
and they will be available to your xarray.Dataset or xarray.DataArray objects through the `spec` attribute, *e.g.*
```python
darr.spec.hs()
```
## Requirements
[SpecArray](https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py#L31) methods require the following attributes in [xarray.DataArray](http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html):
- (required) wave frequency coordinate in Hz named as `freq`
- (optional) wave direction coordinate in degree (coming from) named as `dir`
- wave spectra data in m2/Hz/degree

[SpecDataset](https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py#L16) methods require the following attributes in [xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html):
- (required) DataArray named as `efth` complying to the specifications above

SpecDataset provides a wrapper around the methods in SpecArray accessor. For example, assuming an xarray.Dataset object dset containing an `efth` variable in it, both of these would produce the same result:
```python
dset.efth.spec.hs()
dset.spec.hs()
```

### Example to define and plot spectra history from SWAN bnd spectra file:
```python
from spectra import read_swan

dset = read_swan('/source/pyspectra/tests/antf0.20170207_06z.bnd.swn')
spec_hist = dset.efth.isel(site=0).spec.oned().T
spec_hist.plot()
```
