.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

=======================
Migrating from v3 to v4
=======================

Version 4 brought some backwards-incompatible changes. This page summarises what
needs updating in code written against wavespectra 3.x.

Partitioning
------------

The single ``spec.partition()`` method from v3 was replaced by the ``spec.partition``
namespace, which provides several partitioning techniques following the WAVEWATCH III
naming convention (see :doc:`partitioning`). The previous watershed behaviour is
provided by the :py:meth:`~wavespectra.partition.partition.Partition.ptm1` method:

.. code:: python

    # wavespectra 3.x
    dspart = dset.spec.partition(dset.wspd, dset.wdir, dset.dpt, swells=3)

    # wavespectra 4.x
    dspart = dset.spec.partition.ptm1(dset.wspd, dset.wdir, dset.dpt, swells=3)

Please note:

* The default number of levels used to discretise the spectra in the watershed
  algorithm changed from ``ihmax=200`` in v3 to ``ihmax=100`` in v4. Specify
  ``ihmax=200`` to reproduce previous defaults.
* The underlying watershed algorithm was rewritten from Fortran to C in v4. The two
  implementations produce identical partition maps (versions >= 4.5.2; earlier 4.x
  releases could return incorrect partitions for datasets whose spectral dimensions
  are not the last, trailing dimensions in memory).
* The v3 method applied an extra splitting step based on frequency inflections of the
  watershed partitions which is not implemented in v4, so partition statistics can
  legitimately differ for some spectra.

Plotting
--------

The ``as_log10`` option to plot the logarithm of the energy density was deprecated.
Calculate the logarithm beforehand instead:

.. code:: python

    # wavespectra 3.x
    dset.spec.plot(as_log10=True)

    # wavespectra 4.x
    np.log10(dset.efth).spec.plot(normalised=False)

Reading files
-------------

Reading functions are unchanged, but v4 also registers xarray backend engines for all
readers so files can be opened directly with ``xr.open_dataset``:

.. code:: python

    dset = xr.open_dataset("spectra.nc", engine="ww3")

Construction
------------

The spectral construction and reconstruction functionality was redesigned in v4 with
parametric shape functions defined in the :py:mod:`wavespectra.construct` subpackage
(see :doc:`construction` and :doc:`reconstruction`).
