Output
------

.. py:module:: wavespectra.output

Functions are available to write wavespectra datasets to disk in a number of different
file types. Output functions are defined in modules within the :py:mod:`wavespectra.output`
subpackage, for example, :py:func:`wavespectra.output.swan.to_swan`. They are accessed
as methods in the SpecDataset accessor, for instance:

.. ipython:: python

    dset.spec.to_octopus("_static/octopusfile.oct")

    !head -n 20 _static/octopusfile.oct


The following conventions are expected for defining output functions:

* Output functions must be defined in modules within the `wavespectra.output`_ subpackage.

* Modules should be named as `filetype`.py, e.g., ``swan.py``.

* Functions should be named as to_`filetype`, e.g., ``to_swan``.

* Function **must** accept ``self`` as the first input argument so they can be plugged
  in as methods in the :py:class:`~wavespectra.specdataset.SpecDataset` accessor class.

Available writers
~~~~~~~~~~~~~~~~~

These output functions are currently available as methods of :py:class:`~wavespectra.specdataset.SpecDataset`:

.. currentmodule:: wavespectra

.. autosummary::
    :nosignatures:
    :toctree: generated/

    SpecDataset.to_funwave
    SpecDataset.to_json
    SpecDataset.to_netcdf
    SpecDataset.to_octopus
    SpecDataset.to_orcaflex
    SpecDataset.to_swan
    SpecDataset.to_ww3

.. _wavespectra.output: https://github.com/wavespectra/wavespectra/tree/master/wavespectra/output
