.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

================
Conventions
================
Wavespectra takes advantage of `xarray`_'s labelled coordinates to abstract
n-dimensional wave spectra datasets. This requires some conventions around special
coordinates and variables. The convention adopted is based on that of the NetCDF
output from the WAVEWATCH III spectral wave model.

Datasets and DataArrays that follow these naming and units conventions gain the
``spec`` accessor with the full wavespectra functionality. The table below summarises
the special names:

.. list-table::
   :header-rows: 1
   :widths: 12 12 20 56

   * - Name
     - Type
     - Units
     - Description
   * - ``efth``
     - variable
     - :math:`m^{2}/Hz/degree` (2D), :math:`m^{2}/Hz` (1D)
     - Wave energy density :math:`E(f,\theta)` or :math:`E(f)`; the core spectral
       variable, required by all methods.
   * - ``freq``
     - coordinate
     - :math:`Hz`
     - Wave frequency, required by all methods.
   * - ``dir``
     - coordinate
     - :math:`degree`
     - Wave direction in coming-from convention; required for 2D spectra and
       directional methods.
   * - ``time``
     - coordinate
     -
     - Times of the spectra, required by a few methods such as partition tracking.
   * - ``site``
     - coordinate
     -
     - Index of spectral sites; used by the selecting functionality.
   * - ``lon``, ``lat``
     - coordinate / variable
     - :math:`degree`
     - Longitude and latitude of the spectral sites; required by the selecting
       functionality.
   * - ``wspd``
     - variable
     - :math:`m s^{-1}`
     - Wind speed, required by the partitioning methods that split sea and swell
       (``ptm1``, ``ptm2``, ``ptm4``, ``hp01``).
   * - ``wdir``
     - variable
     - :math:`degree`
     - Wind direction in coming-from convention, required by the same partitioning
       methods as ``wspd``.
   * - ``dpt``
     - variable
     - :math:`m`
     - Water depth, required by the sea/swell partitioning methods and by
       wavenumber-based methods such as
       :py:meth:`~wavespectra.SpecArray.celerity` and
       :py:meth:`~wavespectra.SpecArray.wavelen`.

Directions
----------

Directional quantities (``dir``, ``wdir`` and derived parameters such as ``dm`` and
``dpm``) follow the nautical, *coming-from* convention: the direction waves (or wind)
come from, measured clockwise from true North, in degrees. For example, waves with
``dir=0`` travel southwards and waves with ``dir=270`` come from the West.

Attributes
----------

Pre-defined names and units for these and other coordinates and variables are
provided in the :py:mod:`wavespectra.core.attributes` module:

.. ipython:: python

    from wavespectra.core.attributes import attrs

    attrs.SPECNAME

    attrs.ATTRS.hs


.. _xarray: https://docs.xarray.dev/en/stable/
