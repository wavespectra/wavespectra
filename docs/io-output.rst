.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

======
Output
======

.. py:module:: wavespectra.output

.. ipython:: python
    :suppress:

    from wavespectra import read_ww3
    dset = read_ww3("_static/ww3file.nc")

Functions are available to write wavespectra datasets to disk in a number of different
file types. Output functions are defined in modules within the :py:mod:`wavespectra.output`
subpackage, for example, :py:func:`wavespectra.output.swan.to_swan`. They are accessed
as methods in the SpecDataset accessor, for instance:

.. ipython:: python

    dset.spec.to_octopus("octopusfile.oct")

    !head -n 20 octopusfile.oct


The following conventions are expected for defining output functions:

* Output functions must be defined in modules within the `wavespectra.output`_ subpackage.

* Modules should be named as `filetype`.py, e.g., ``swan.py``.

* Functions should be named as to_`filetype`, e.g., ``to_swan``.

* Functions **must** accept ``self`` as the first input argument so they can be plugged
  in as methods in the :py:class:`~wavespectra.specdataset.SpecDataset` accessor class.

Available writers
-----------------

These output functions are currently available as methods of :py:class:`~wavespectra.specdataset.SpecDataset`:

.. currentmodule:: wavespectra

.. autosummary::
    :nosignatures:
    :toctree: generated/

    SpecDataset.to_funwave
    SpecDataset.to_funwave_new
    SpecDataset.to_json
    SpecDataset.to_netcdf
    SpecDataset.to_octopus
    SpecDataset.to_orcaflex
    SpecDataset.to_swan
    SpecDataset.to_ww3

.. _wavespectra.output: https://github.com/wavespectra/wavespectra/tree/main/wavespectra/output
