.. image:: _static/wavespectra_logo.png
   :width: 150 px
   :align: right

Output
======

.. py:module:: wavespectra.output

Functions to write :py:class:`~wavespectra.specdataset.SpecDataset` into files.

The output functions are attached as methods in the SpecDataset accessor. They
are should be created in modules within :py:mod:`wavespectra.output` subpackage
so they can be dynamically plugged as SpecDataset methods.

The following convention is expected for defining output functions:

- Funcions for different file types are defined in different modules within
  :py:mod:`wavespectra.output` subpackage.
- Modules are named as `filetype`.py, e.g., ``swan.py``.
- Functions are named as to_`filetype`, e.g., ``to_swan``.
- Function **must** accept ``self`` as the first input argument.

The ouput functions are described in the
:py:class:`~wavespectra.specdataset.SpecDataset` section.