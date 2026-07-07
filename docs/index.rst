.. image:: _static/wavespectra_logo.png
   :width: 150 px
   :align: right

wavespectra
===========

Python library for wave spectra
-------------------------------

Wavespectra is an open source project for working with ocean wave spectral data hosted
on https://github.com/wavespectra/wavespectra. The library is built on top of
`xarray`_, leveraging xarray's labelled multi-dimensional arrays and making dealing
with wave spectra simple and fast. It provides readers for 20+ file formats from
spectral wave models and observation platforms, 60+ methods to calculate integrated
wave parameters and manipulate spectra, spectral partitioning, parametric spectral
construction and reconstruction, polar plotting and writers for several spectral
file formats.

.. code:: python

    import xarray as xr
    import wavespectra

    dset = xr.open_dataset("spectra.nc", engine="ww3")
    hs = dset.spec.hs()
    dspart = dset.spec.partition.ptm1(dset.wspd, dset.wdir, dset.dpt)
    dset.isel(time=0, site=0).spec.plot(kind="contourf")

.. _xarray: https://xarray.pydata.org/en/stable/

Documentation
-------------

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    install
    quickstart
    conventions
    io-input
    io-output
    plotting
    selecting
    partitioning
    construction
    reconstruction
    cli

.. toctree::
    :maxdepth: 1
    :caption: Help & Reference:

    api
    migration
    gallery
    support
    contributing
    history
    authors

History
-------
Wavespectra started as an internal tool developed by Rafael Guedes, Tom Durrant and
David Johnson at `Metocean Solutions`_. It was released as open source in April
2018 and received contributions from Jorge Perez. The project was transitioned into a fully
community-developed project in July 2019, hosted at the `wavespectra`_ github organisation.


.. _`Metocean Solutions`: https://www.metocean.co.nz/
.. _`wavespectra`: https://github.com/wavespectra

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
