.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

================
Conventions
================
Wavespectra takes advantage of `xarray`_'s labelled coordinates to abstract
n-dimensional wave spectra datasets and calculate integrated spectral parameters.
This requires some conventions around special coordinates and variables.
This naming convention is inspired on netcdf files from WAVEWATCH III wave model.

Coordinates
-----------

Wave frequency
~~~~~~~~~~~~~~

Wave frequency coordinate named as ``freq``, defined in :math:`Hz` (required).

Wave direction
~~~~~~~~~~~~~~

Wave direction coordinate in coming-from convention, named as ``dir``,
defined in :math:`degree` (required for 2D spectra and directional methods).

Time
~~~~
Time coordinate named as ``time`` (required by some methods).

Data variables
--------------

Wave energy density
~~~~~~~~~~~~~~~~~~~
Wave energy density array named as ``efth``, defined in:

* 2D spectra :math:`E(\sigma,\theta)`: :math:`m^{2}{degree^{-1}}{s}`.
* 1D spectra :math:`E(\sigma)`: :math:`m^{2}{s}`.

Wind speed
~~~~~~~~~~
Wind speed array named as ``wspd``, defined in :math:`ms^{-1}` (required for watershed partitioning).

Wind direction
~~~~~~~~~~~~~~
Wind direction array named as ``wdir``, defined in :math:`degree` (required for watershed partitioning).

Water depth
~~~~~~~~~~~
Water depth array named as ``dpt``, defined in :math:`m` (required for watershed partitioning and wavenumber-based methods).

Attributes
----------

Pre-defined names and units for these and other coordintes and variables are
available from module :py:mod:`wavespectra.core.attributes`. This module defines
variable names and some CF attributes by loading information from
`attributes.yml`_ file. The attributes can be accessed for example as:

.. ipython:: python

    from wavespectra.core.attributes import attrs

    attrs.SPECNAME

    attrs.ATTRS.hs

The module also provides a function to standarise coordinate and variable
attributes in a Dataset object using the information defined in `attributes.yml`_:

.. autofunction:: wavespectra.core.attributes.set_spec_attributes
   :noindex:

.. _xarray: https://xarray.pydata.org/en/stable/
.. _attributes.yml: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/core/attributes.yml
