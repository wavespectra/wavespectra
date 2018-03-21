Wavespectra
===========
**Python library for ocean wave spectra.**

Wavespectra is a library built on top of `xarray`_ for reading, manipulating,
processing, and writing ocean wave spectra data.

The code is based on two `accessor classes`_:

- :py:class:`wavespectra.specarray.SpecArray`: extends `xarray.DataArray`_ to
  manipulate wave spectra and calculate integrated spectral parameters.
- :py:class:`wavespectra.specdataset.SpecDataset`: extends `xarray.Dataset`_
  with methods for writing spectra in different formats.

These accessors are registered to `xarray.DataArray`_ and `xarray.Dataset`_
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

SpecArray
---------
:py:class:`wavespectra.specarray.SpecArray` extends `xarray.DataArray`_.

SpecArray provides several methods for calculating integrated wave parameters
from wave spectra. For instance, significant wave height can be calculated from
a hypothetical SpecArray object `efth` as:

.. code:: python

    hs = efth.spec.hs()

which returns a `xarray.DataArray`_ object `hs`.

The following attributes are required when defining SpecArray object:

- Wave frequency coordinate in `Hz`, named as `freq`.
- Wave direction coordinate (if 2D spectra) in `degree` "coming_from", named as
  `dir`.
- Wave energy density array in `m^{2}.s.degree^{-1}`, named as `efth`.

The following methods are available in SpecArray:

.. autoclass:: wavespectra.specarray.SpecArray
   :members:

SpecDataset
-----------
:py:class:`wavespectra.specdataset.SpecDataset` extends `xarray.Dataset`_.

SpecDataset is useful for storing wave spectra alongside other variables that
share some common dimensions. It provides methods for writing wave spectra into
different file formats.

SpecDataset works as a wrapper around
:py:class:`wavespectra.specarray.SpecArray`. All public methods from SpecArray
can be directly accessed from SpecDataset. For instance, these two calls are
equivalent:

.. code:: python

    # Calling hs method from SpecArray
    hs = dset.efth.spec.hs()
    # Calling hs method from SpecDataset
    hs = dset.spec.hs()

both cases return identical `xarray.DataArray`_ objects `hs`.

The following methods are available in SpecDataset:

.. autoclass:: wavespectra.specdataset.SpecDataset
   :members:
   :exclude-members: to_datetime

Input
-----

.. py:module:: wavespectra.input

Functions to read wave spectra from file into
:py:class:`wavespectra.specdataset.SpecDataset`.

The input functions allow abstracting away the format the wave spectra data are
stored on disk and loading them into a standard SpecDataset object. The methods
for calculating integrated spectral parameters and writing spectra as different
file formats become available from the `spec` namespece.

Reading functions are defined in modules within
:py:mod:`wavespectra.input` subpackage. The functions are imported at the main
module level and can be accessed for instance as:

.. code:: python

    from wavespectra import read_swan

    dset = read_swan('my_swan_file')

The following convention is expected for defining reading functions:

- Funcions for different file types are defined in different modules within
  :py:mod:`wavespectra.input` subpackage.
- Modules are named as <`filetype`>.py, e.g., ``swan.py``.
- Functions are named as read_<`filetype`>, e.g., ``read_swan``.

Input functions can also be defined without following this convention. However
they are not accessible from the main module level and need to be imported from
their full module path, for instance:

.. code:: python

    from wavespectra.input.swan import read_hotswan

    dset = read_hotswan('my_swan_hotfiles')

The following input functions are currently availeble from the main module
level:

NETCDF
~~~~~~

.. autofunction:: wavespectra.read_netcdf

SWAN
~~~~

.. autofunction:: wavespectra.read_swan

WW3
~~~

.. autofunction:: wavespectra.read_ww3

WW3-MSL
~~~~~~~

.. autofunction:: wavespectra.read_ww3_msl

OCTOPUS
~~~~~~~

.. autofunction:: wavespectra.read_octopus

JSON
~~~~

.. autofunction:: wavespectra.read_json

Other functions
~~~~~~~~~~~~~~~
These functions are not accessible from the main module level and need to be
imported from their full module path:

.. autofunction:: wavespectra.input.swan.read_hotswan
.. autofunction:: wavespectra.input.swan.read_swans
.. autofunction:: wavespectra.input.swan.read_swanow

Output
------

Functions to write :py:class:`wavespectra.specdataset.SpecDataset` into files.

The output functions are attached as methods in the SpecDataset accessor. They
are defined in modules within :py:mod:`wavespectra.output` subpackage and are
dynamically plugged as SpecDataset methods.

The following convention is expected for defining output functions:

- Funcions for different file types are defined in different modules within
  :py:mod:`wavespectra.output` subpackage.
- Modules are named as <`filetype`>.py, e.g., ``swan.py``.
- Functions are named as to_<`filetype`>, e.g., ``to_swan``.
- Function **must** accept ``self`` as the first input argument.

The ouput functions are described in :py:class:`wavespectra.specdataset.SpecDataset`.

.. _xarray: https://xarray.pydata.org/en/stable/
.. _SpecArray: https://github.com/metocean/wavespectra/blob/master/wavespectra/specarray.py
.. _SpecDataset: https://github.com/metocean/wavespectra/blob/master/wavespectra/specdataset.py
.. _xarray.DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _`accessor classes`: http://xarray.pydata.org/en/stable/internals.html?highlight=accessor