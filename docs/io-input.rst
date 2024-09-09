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
into :py:class:`~wavespectra.specdataset.SpecDataset` objects. The functions are
defined in modules within the :py:mod:`wavespectra.input` subpackage, for example,
:py:func:`wavespectra.input.swan.read_swan`. They can be imported from the main module
level for convenience, for instance:

.. ipython:: python

    from wavespectra import read_swan

    dset = read_swan("_static/swanfile.spec")
    dset.spec

The following conventions are expected for defining fully-supported reading functions:

* Input functions must be defined in modules within the `wavespectra.input`_ subpackage.

* Modules should be named as `filetype`.py, e.g., ``swan.py``.

* Functions should be named as read_`filetype`, e.g., ``read_swan``.

Input functions can also be defined without following these conventions. However
they will not be accessible from the main module level and will have to be imported
from their full module path, e.g.

.. ipython:: python

    from wavespectra.input.swan import read_hotswan

Backends engines
~~~~~~~~~~~~~~~~

Backend engines are provided to seamlessly access wavespectra datasets through the
`xarray.open_dataset`_ function. These engines are designated by their respective
``filetype``, for example:

.. ipython:: python

    import xarray as xr

    dset = xr.open_dataset("_static/ww3file.nc", engine="ww3")
    dset.spec

Available readers
~~~~~~~~~~~~~~~~~

These input functions are currently available from the main module level:

.. currentmodule:: wavespectra

.. autosummary::
    :nosignatures:
    :toctree: generated/

    read_awac
    read_dataset
    read_datawell
    read_era5
    read_funwave
    read_json
    read_ncswan
    read_ndbc
    read_ndbc_ascii
    read_netcdf
    read_obscape
    read_octopus
    read_spotter
    read_swan
    read_triaxys
    read_ww3
    read_ww3_station
    read_wwm
    read_xwaves

These functions are not accessible from the main module level and need to be
imported from their full module path:

.. autosummary::
   :nosignatures:
   :toctree: generated/

   input.swan.read_swans
   input.swan.read_hotswan
   input.swan.read_swanow

.. note::

    Wavespectra supports different data formats including ``NetCDF``, ``ASCII``,
    ``JSON``. netcdf-type datasets, i.e. those that can be open with xarray's
    open_dataset_ and open_mfdataset_ functions, can also be prescribed in ``ZARR``
    format (wavespectra uses xarray's open_zarr_ function behind the scenes to open
    these files). Functions that support ZARR have a ``file_format`` argument option.

    Files in ``ZARR`` format can be open from both local and remote (bucket) stores.
    Files in ``NetCDF`` format can be open from remote (i.e., bucket) stores when they
    are opened using ``fsspec``.

.. _open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
.. _open_mfdataset: https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
.. _open_zarr: https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html
.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
.. _wavespectra.input: https://github.com/wavespectra/wavespectra/tree/master/wavespectra/input

.. include:: io-output.rst
