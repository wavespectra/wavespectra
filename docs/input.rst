.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

==============
Input & Output
==============

Input
-----

.. py:module:: wavespectra.input

Wavespectra provides several functions for reading datasets from different file formats
into :py:class:`~wavespectra.specdataset.SpecDataset` objects. The functions are defined
in submodules within :py:mod:`wavespectra.input`. They can be imported from the main
module level for convenience, for instance:

.. ipython:: python

    from wavespectra import read_swan

The following convention is expected for defining reading functions:

- Funcions for different file types are defined in different modules within
  :py:mod:`wavespectra.input` subpackage.
- Modules are named as `filetype`.py, e.g., ``swan.py``.
- Functions are named as read_`filetype`, e.g., ``read_swan``.

Input functions can also be defined without following this convention. However
they are not accessible from the main module level and need to be imported from
their full module path, e.g.

.. ipython:: python

    from wavespectra.input.swan import read_hotswan

These input functions are currently available from the main module level:

WW3 NetCDF file
~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_ww3
   :noindex:

NCSWAN NetCDF file
~~~~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_ncswan
   :noindex:

WWM NetCDF file
~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_wwm
   :noindex:

Generic NetCDF file
~~~~~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_netcdf
   :noindex:

xarray DATASET
~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_dataset
   :noindex:

SWAN ASCII file
~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_swan
   :noindex:

TRIAXYS ASCII file
~~~~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_triaxys
   :noindex:

SPOTTER ASCII file
~~~~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_spotter
   :noindex:

OCTOPUS ASCII file
~~~~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.read_octopus
   :noindex:

These functions are not accessible from the main module level and need to be
imported from their full module path:

SWAN hot files
~~~~~~~~~~~~~~

.. autofunction:: wavespectra.input.swan.read_hotswan
   :noindex:

SWAN ASCII glob files
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: wavespectra.input.swan.read_swans
   :noindex:

SWAN ASCII nowcast
~~~~~~~~~~~~~~~~~~
.. autofunction:: wavespectra.input.swan.read_swanow
   :noindex:


.. include:: output.rst