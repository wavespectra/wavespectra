Wavespectra
===========
**Python library for ocean wave spectra.**

Wavespectra is a library built on top of `xarray`_ for reading, manipulating,
processing, and writing ocean wave spectra data.

The code is based on two main classes:

- :py:class:`wavespectra.specarray.SpecArray`: object based on xarray.`DataArray`_ to manipulate wave spectra
  and calculate integrated spectral parameters.
- SpecDataset_: object based on xarray.`Dataset`_ with methods for writing
  spectra in different formats.

SpecArray
---------

SpecDataset
-----------

Input
-----

Output
------

.. _xarray: https://xarray.pydata.org/en/stable/
.. _SpecArray: https://github.com/metocean/wavespectra/blob/master/wavespectra/specarray.py
.. _SpecDataset: https://github.com/metocean/wavespectra/blob/master/wavespectra/specdataset.py
.. _DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html