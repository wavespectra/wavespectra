.. image:: _static/Gradient_wide.png
   :width: 150 px
   :align: right

================
Conventions
================
Wavespectra takes advantage of `xarray`_'s labelled coordinates to abstract
N-dimensional wave spectra and calculate integrated spectral parameters. This
requires some conventions around special coordinates and variables:

- Wave frequency coordinate named as ``freq``, defined in :math:`Hz`.
- Wave direction coordinate named as ``dir``, defined in :math:`degree`,
  (coming_from).
- Wave energy density array named as ``efth``, defined in:
    - 2D spectra :math:`E(\sigma,\theta)`: :math:`m^{2}{degree^{-1}}{s}`.
    - 1D spectra :math:`E(\sigma)`: :math:`m^{2}{s}`.
- Time coordinate named as ``time``.

Attributes
----------

Pre-defined names and units for these and other coordintes and variables are
available from :py:mod:`wavespectra.core.attributes`. This module defines
variable names and some CF attributes by loading information from
`attributes.yml`_ file. The attributes can be accessed for example as:

.. code:: python

    from wavespectra.core.attributes import attrs

    In [1]: attrs.SPECNAME
    Out[1]: 'efth'

    In [2]: attrs.ATTRS.hs
    Out[2]: AttrDict({'units': 'm', 'standard_name': 'sea_surface_wave_significant_height'})

The module also provides a function to standarise coordinate and variable
attributes in a Dataset object using the information defined in `attributes.yml`_:

.. autofunction:: wavespectra.core.attributes.set_spec_attributes
   :noindex:

.. _xarray: https://xarray.pydata.org/en/stable/
.. _attributes.yml: https://github.com/metocean/wavespectra/blob/master/wavespectra/core/attributes.yml