.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

================
Conventions
================
Wavespectra takes advantage of `xarray`_'s labelled coordinates to abstract
n-dimensional wave spectra datasets. This requires some conventions around special
coordinates and variables. The convention adopted is based on that of the netcdf
output from the WAVEWATCH III spectral wave model.

Coordinates
-----------

Wave frequency
~~~~~~~~~~~~~~

Wave frequency coordinate with name ``freq``, defined in :math:`Hz` (required).

Wave direction
~~~~~~~~~~~~~~

Wave direction coordinate in coming-from convention, with name ``dir``,
defined in :math:`degree` (required for 2D spectra and directional methods).

Time
~~~~
Time coordinate with name ``time`` (required by a few methods).

Data variables
--------------

Wave energy density
~~~~~~~~~~~~~~~~~~~
Wave energy density array named with name ``efth``, defined in:

* 2D spectra :math:`E(\sigma,\theta)`: :math:`m^{2}{degree^{-1}}{s}`.
* 1D spectra :math:`E(\sigma)`: :math:`m^{2}{s}`.

Wind speed
~~~~~~~~~~
Wind speed array named with name ``wspd``, defined in :math:`ms^{-1}` (required for the
watershed partitioning).

Wind direction
~~~~~~~~~~~~~~
Wind direction array with ``wdir``, defined in :math:`degree` (required for the
watershed partitioning).

Water depth
~~~~~~~~~~~
Water depth array with name ``dpt``, defined in :math:`m` (required for the watershed
partitioning and wavenumber-based methods).

Attributes
----------

Pre-defined names and units for these and other coordinates and variables are
provided in the :py:mod:`wavespectra.core.attributes` module:

.. ipython:: python

    from wavespectra.core.attributes import attrs

    attrs.SPECNAME

    attrs.ATTRS.hs


.. _xarray: https://docs.xarray.dev/en/stable/
