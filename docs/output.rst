Output
------

.. py:module:: wavespectra.output

Wavespectra provides functions to write :py:class:`~wavespectra.specdataset.SpecDataset`
objects into different file types. They are defined in module within :py:mod:`wavespectra.output`.
They are attached as methods in the SpecDataset accessor.

The following convention is expected for defining output functions:

- Funcions for different file types are defined in different modules within
  :py:mod:`wavespectra.output` subpackage.
- Modules are named as `filetype`.py, e.g., ``swan.py``.
- Functions are named as to_`filetype`, e.g., ``to_swan``.
- Function **must** accept ``self`` as the first input argument.

These output functions are currently available as methods of :py:class:`~wavespectra.specdataset.SpecDataset`:

Generic NetCDF file
~~~~~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.output.netcdf.to_netcdf
   :noindex:

SWAN ASCII file
~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.output.swan.to_swan
   :noindex:

OCTOPUS ASCII file
~~~~~~~~~~~~~~~~~~~

.. autofunction:: wavespectra.output.octopus.to_octopus
   :noindex:
