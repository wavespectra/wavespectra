.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

============
Partitioning
============

A new partitioning api has been implemented in Wavespectra version 4. Previously, two
methods were available for partitioning in :class:`~wavespectra.specarray.SpecArray`,
the `split` method for a simple thresholds-based, frequency split, and the `partition`
method for the watershed partitioning based on `Hanson et al. (2008)`_ and implemented
in spectral wave models such as WW3 and SWAN.

In Wavespectra version 4, :meth:`~wavespectra.specarray.SpecArray.partition` became a
new namespace from which the several partitioning methods can be accessed.



.. ipython:: python
    :okexcept:
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import read_ww3, read_swan
    from wavespectra.construct.frequency import pierson_moskowitz, jonswap, tma, gaussian
    from wavespectra.construct.direction import cartwright, asymmetric
    from wavespectra.construct import construct_partition, partition_and_reconstruct
    freq = np.arange(0.03, 0.401, 0.001)
    dir = np.arange(0, 360, 1)


One-dimensional fit
___________________




.. _`Hanson et al. (2008)`: https://journals.ametsoc.org/view/journals/atot/26/8/2009jtecho650_1.xml
