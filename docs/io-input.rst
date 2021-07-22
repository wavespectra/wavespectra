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

.. note::

    The following conventions are expected for defining reading functions:

    - Funcions for different file types are defined in different modules within :py:mod:`wavespectra.input` subpackage.

    - Modules are named as `filetype`.py, e.g., ``swan.py``.

    - Functions are named as read_`filetype`, e.g., ``read_swan``.

Input functions can also be defined without following these conventions. However
they are not accessible from the main module level and need to be imported from
their full module path, e.g.

.. ipython:: python

    from wavespectra.input.swan import read_hotswan

.. note::

    Wavespectra currently supports data formats that include ``NetCDF``, ``ASCII``
    and ``JSON`` type files. NetCDF type datasets, i.e. those that can be open with
    xarray's open_dataset_ and open_mfdataset_ functions, can also be prescribed in ``ZARR``
    format (wavespectra uses xarray's open_zarr_ function behind the scenes to open
    these files). Functions that support ZARR have a ``file_format`` argument option.

.. note::

    Files in ``ZARR`` format can be open from both local and remote (bucket) stores.

.. _open_dataset: https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html#xarray.open_dataset
.. _open_mfdataset: https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html#xarray.open_mfdataset
.. _open_zarr: https://xarray.pydata.org/en/stable/generated/xarray.open_zarr.html#xarray.open_zarr


These input functions are currently available from the main module level:

.. currentmodule:: wavespectra

.. autosummary::
    :nosignatures:
    :toctree: generated/

    read_ww3
    read_ncswan
    read_wwm
    read_netcdf
    read_era5
    read_swan
    read_triaxys
    read_ndbc
    read_spotter
    read_octopus
    read_dataset
    read_json
    read_funwave


These functions are not accessible from the main module level and need to be
imported from their full module path:

.. autosummary::
   :nosignatures:
   :toctree: generated/

   input.swan.read_swans
   input.swan.read_hotswan
   input.swan.read_swanow


.. include:: io-output.rst