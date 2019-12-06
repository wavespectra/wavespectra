.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

=========
Selecting
=========

Wavespectra complements xarray's selecting_ and interpolating_ functionality with functions to select and
interpolate from ``site`` (1D) coordinates. The functions are defined in :py:mod:`wavespectra.core.select`
module and can be accessed via the :py:meth:`~wavespectra.specdataset.SpecDataset.sel` method from
:py:class:`~wavespectra.specarray.SpecArray` and :py:class:`~wavespectra.specdataset.SpecDset` accessors.

Nearest neighbour
-----------------
Select from nearest sites.

.. ipython:: python

    from wavespectra import read_ww3
    dset = read_ww3("_static/ww3file.nc")
    ds = dset.spec.sel(
        lons=[92.01, 92.05, 92.09],
        lats=[19.812, 19.875, 19.935],
        method="nearest"
    )
    ds


Inverse distance weighting
--------------------------
Interpolate at exact locations via inverse distance weighting algorithm.

.. ipython:: python

    ds = dset.spec.sel(
        lons=[92.01, 92.05, 92.09],
        lats=[19.812, 19.875, 19.935],
        method="idw"
    )
    ds

Bounding box
------------
Select all sites withing bounding box.

.. ipython:: python

    ds = dset.spec.sel(
        lons=[91, 93],
        lats=[19, 20],
        method="bbox"
    )
    ds


.. note::

    When working with large datasets with thousands of spectra sites, it is
    recommended using `chunks={"site": 1}` option to open dataset lazily in an efficient
    way for selecting sites. The downside is that accessing entire site-dependent variables
    (notably ``lon`` and ``lat``) becomes slower, affecting the performance of selecting
    functions. This can be circumvented by loading these variables without the
    `chunks` options, and using them as arguments in ``sel``, e.g.

.. ipython:: python

    coords = read_ww3("_static/ww3file.nc")[["lon", "lat"]]
    dset = read_ww3("_static/ww3file.nc", chunks={"site": 1})
    ds = dset.spec.sel(
        lons=[92.01, 92.05, 92.09],
        lats=[19.812, 19.875, 19.935],
        method="idw",
        dset_lons=coords.lon,
        dset_lats=coords.lat
    )
    ds

.. _selecting: https://xarray.pydata.org/en/latest/indexing.html
.. _interpolating: https://xarray.pydata.org/en/latest/interpolation.html
