.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

=========
Selecting
=========

Wavespectra complements xarray's selecting_ and interpolating_ functionality with functions to select and
interpolate from ``site`` (1D) coordinates. The functions are defined in :py:mod:`wavespectra.core.select`
module and can be accessed via the :py:meth:`~wavespectra.specdataset.SpecDataset.sel` method from the
:py:class:`~wavespectra.specarray.SpecArray` and :py:class:`~wavespectra.specdataset.SpecDataset` accessors.

.. note::

    The select methods in wavespectra are designed to work with 1D spatial coordinates
    only which is typically the case for spectral data. For 2D, gridded coordinates,
    use xarray's native indexing and interpolation methods.


Nearest neighbour
-----------------
Select from nearest sites.

.. ipython:: python
    :okwarning:

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
    :okwarning:

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
    :okwarning:

    ds = dset.spec.sel(
        lons=[91, 93],
        lats=[19, 20],
        method="bbox"
    )
    ds


.. _selecting: https://xarray.pydata.org/en/latest/indexing.html
.. _interpolating: https://xarray.pydata.org/en/latest/interpolation.html
