===========
Wavespectra
===========
**Python library for ocean wave spectra.**

Wavespectra is a library for processing ocean wave spectral data. It provides
reading and writing of common spectral data formats, calculation of common
integrated wave paramaters, several methods of spectral partitioning and
spectral manipulation in a package focussed on speed and efficiency for large
numbers of spectra.

The library built on top of `xarray`_. The extended functionality is based on
two `accessor classes`_:

- :py:class:`~wavespectra.specarray.SpecArray`: extends xarray's `DataArray`_
  with methods to manipulate wave spectra and calculate integrated spectral
  parameters.
- :py:class:`~wavespectra.specdataset.SpecDataset`: extends xarray's `Dataset`_
  with methods for writing spectra in different formats.

These accessors are registered to `DataArray`_ and `Dataset`_
objects as a new namespece called ``spec``. To attach the namespace simply import
the accessors into your code, e.g.:

.. code:: python

    import xarray as xr
    import numpy as np
    import datetime

    from wavespectra.specarray import SpecArray

    coords = {'time': [datetime.datetime(2017,01,n+1) for n in range(2)],
              'freq': [0.05,0.1],
              'dir': np.arange(0,360,120)}
    efth = xr.DataArray(data=np.random.rand(2,2,3),
                        coords=coords,
                        dims=('time','freq', 'dir'),
                        name='efth')

    In [1]: efth.spec.hs()
    Out[1]:
    <xarray.DataArray 'hs' (time: 2)>
    array([ 10.128485,   9.510618])
    Coordinates:
    * time     (time) datetime64[ns] 2017-01-01 2017-01-02
    Attributes:
        standard_name: sea_surface_wave_significant_height
        units: m

Using accessors in this way provides an elegant way to extend and leverage the
power of `xarray`_.  In common usage, this mechanism is largely transparent,
with the provided reading functions , returning the above classes from which
the spectral methods can be directly accessed (see :ref:`Input`).  For example,
to read a swan spectral file and calculate the corresponing significant wave
heights: 

.. code:: python

    from wavespectra import read_swan

    dset = read_swan('my_swan_file')
    hs = dset.spec.hs()


Details of each class are below. 

SpecArray
---------
:py:class:`~wavespectra.specarray.SpecArray` extends `DataArray`_.

SpecArray provides several methods for calculating integrated wave parameters
from wave spectra. For instance, significant wave height can be calculated from
a hypothetical SpecArray object `efth` as:

.. code:: python

    hs = efth.spec.hs()

which returns a `DataArray`_ object `hs`.

The following attributes are required when defining SpecArray object:

- Wave frequency coordinate named as ``freq``, defined in :math:`Hz`.
- Wave direction coordinate (if 2D spectra) named as ``dir``, defined in
  :math:`degree`, (coming_from).
- Wave energy density array named as ``efth``, defined in:
    - 2D spectra :math:`E(\sigma,\theta)`: :math:`m^{2}{degree^{-1}}{s}`.
    - 1D spectra :math:`E(\sigma)`: :math:`m^{2}{s}`.
- Time coordinate (if present) named as ``time``.

The following methods are available in
:py:class:`~wavespectra.specarray.SpecArray`:

.. autoclass:: wavespectra.specarray.SpecArray
   :members:
   :noindex:

SpecDataset
-----------
:py:class:`~wavespectra.specdataset.SpecDataset` extends `Dataset`_.

SpecDataset is useful for storing wave spectra alongside other variables that
share some common dimensions. It provides methods for writing wave spectra into
different file formats.

SpecDataset works as a wrapper around
:py:class:`~wavespectra.specarray.SpecArray`. All public methods from SpecArray
can be directly accessed from SpecDataset. For instance, these two calls are
equivalent:

.. code:: python

    # Calling hs method from SpecArray
    hs = dset.efth.spec.hs()
    # Calling hs method from SpecDataset
    hs = dset.spec.hs()

both cases return identical `DataArray`_ objects `hs`.

The following methods are available in SpecDataset:

.. autoclass:: wavespectra.specdataset.SpecDataset
   :members:
   :noindex:
   :exclude-members: to_datetime

.. _xarray: https://xarray.pydata.org/en/stable/
.. _SpecArray: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specarray.py
.. _SpecDataset: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specdataset.py
.. _DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _`accessor classes`: http://xarray.pydata.org/en/stable/internals.html?highlight=accessor
