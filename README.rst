=========
pyspectra
=========
Python library for wave spectra

--------------
Main contents:
--------------
- SpecArray_: object based on xarray's `DataArray`_ to manipulate wave spectra and calculate spectral statistics
- SpecDataset_: wrapper around `SpecArray`_ with methods for saving spectra in different formats
- readspec_: access functions to read spectra from different file formats into SpecDataset_ objects

--------------
Code structure
--------------
The two main classes SpecArray_ and SpecDataset_ are defined as `xarray accessors`_. The accessors are registered on xarray's DataArray_ and Dataset_ respectively as a new namespace called `spec`.

.. _SpecArray: https://github.com/metocean/pyspectra/blob/master/spectra/specarray.py
.. _SpecDataset: https://github.com/metocean/pyspectra/blob/master/spectra/specdataset.py
.. _DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _readspec: https://github.com/metocean/pyspectra/blob/master/spectra/readspec.py
.. _xarray accessors: http://xarray.pydata.org/en/stable/internals.html?highlight=accessor