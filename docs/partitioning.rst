.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

============
Partitioning
============

A new partitioning api has been implemented in Wavespectra version 4. Previously, two
methods were available in :class:`~wavespectra.specarray.SpecArray` for spectral
partitioning, the `split` method for threshold-based frequency splits and the
`partition` method for the watershed partitioning based on `Hanson et al. (2008)`_
and implemented in spectral wave models such as WW3 and SWAN.

In version 4, :meth:`~wavespectra.specarray.SpecArray.partition` became a namespace of
the `spec` accessor from which several partitioning methods are now available including:

- :meth:`~wavespectra.partition.partition.Partition.ptm1`
- :meth:`~wavespectra.partition.partition.Partition.ptm1_track`
- :meth:`~wavespectra.partition.partition.Partition.ptm2`
- :meth:`~wavespectra.partition.partition.Partition.ptm3`
- :meth:`~wavespectra.partition.partition.Partition.ptm4`
- :meth:`~wavespectra.partition.partition.Partition.ptm5`
- :meth:`~wavespectra.partition.partition.Partition.hp01`
- :meth:`~wavespectra.partition.partition.Partition.bbox`

The `PTM` methods are named after the convention in the `WAVEWATCHIII`_ spectral wave
model from which they were derived (`PTM1_TRACK` is a modified version of `PTM1` that
tracks and merges spectral partitions in time).

The `HP01` method is an attempt to implement the merging of nearby swell partitions
(in spectral space) described in `Hanson and Phillips (2001)`_. However, the
implementation is still under development and may not work as expected (contribution is
welcome).

The `BBOX` method is a custom method to split the energy
density inside and outside a defined bounding box in spectral space. 


.. ipython:: python
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import read_ww3, read_wwm


PTM1
____

The PTM1 method corresponds to the deprecated `spec.partition()` method from Wavespectra
version 3. In PTM1, topographic partitions for which the percentage of wind-sea energy
exceeds a defined fraction are aggregated and assigned to the wind-sea component (e.g.,
the first partition). The remaining partitions are assigned as swell components in
order of decreasing wave height.

.. ipython:: python
    :okwarning:

    #dset = read_ww3("_static/ww3file.nc")
    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm1(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitionint_ptm1.png
    plt.draw()

Smoothing the spectra before partitioning can help to avoid spurious partitions as
suggested by (`Portilla et al., 2009`_).


.. ipython:: python
    :okwarning:

    dspart = dset.spec.partition.ptm1(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
        smooth=True,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitionint_ptm1_smooth.png
    plt.draw()


Some watershed parameters are exposed to the user for tuning the partitioning algorithm:

.. ipython:: python
    :okwarning:

    dspart = dset.spec.partition.ptm1(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
        agefac=1.5,
        wscut=0.5,
        ihmax=200,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitionint_ptm1_tuning.png
    plt.draw()


PTM2
____

PTM2 works in a similar way to PTM1 by identifying a primary wind sea (assigned to
partition 0) and one or more swell components. In this method however all the swell
partitions are checked for the influence of wind-sea with energy within spectral bins
within the wind-sea range (as defined by a wave age criterion) removed and combined
into a secondary wind-sea partition (assigned to partition 1). The remaining swell
partitions are then assigned in order of decreasing wave height from partition 2 onwards.
This implies PTM2 has an extra partition compared to PTM1.

.. ipython:: python
    :okwarning:

    dspart = dset.spec.partition.ptm2(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitionint_ptm2.png
    plt.draw()


PTM3
____
PTM3 does not classify the topographic partitions into wind-sea or swell - it simply
orders them by wave height. This approach is useful for producing data for spectral
reconstruction applications using a limited number of partitions, where the
classification of the partition as wind-sea or swell is less important than the
proportion of overall spectral energy each partition represents. In addition, this method
does not require wind and water depth information and can be used with any spectral
dataset.

.. ipython:: python
    :okwarning:

    dspart = dset.spec.partition.ptm3(parts=3)
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitionint_ptm3.png
    plt.draw()


PTM4
____
PTM4 uses the wave age criterion derived from the local wind speed to split the spectrum
into a wind-sea and single swell partition. In this case waves with a celerity greater
than the directional component of the local wind speed are considered to be freely
propogating swell (i.e. unforced by the wind). This is similar to the method commonly
used to generate wind-sea and swell from the WAM model.

.. ipython:: python
    :okwarning:

    dspart = dset.spec.partition.ptm4(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        agefac=1.7,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitionint_ptm4.png
    plt.draw()

The wind sea region used to partition the spectra in PTM4 can be calculated
from the :func:`~wavespectra.core.utils.waveage` method:

.. ipython:: python
    :okwarning:

    from wavespectra.core.utils import waveage
    ds = read_ww3("_static/ww3file.nc").sortby("dir").isel(site=0, drop=True)
    windmask = waveage(ds.freq, ds.dir, ds.wspd, ds.wdir, ds.dpt, 1.7)
    f = windmask.fillna(1.0).spec.plot(col="time", col_wrap=3);
    for ind, ax in enumerate(f.axs.flat):
        wdir = float(ds.wdir.isel(time=ind).values)
        ax.set_title(f"wdir={wdir:0.0f} deg")

    @savefig partitioning_windmask.png
    plt.draw()


.. _`WAVEWATCHIII`: https://github.com/NOAA-EMC/WW3
.. _`Hanson and Phillips (2001)`: https://journals.ametsoc.org/view/journals/atot/18/2/1520-0426_2001_018_0277_aaoosd_2_0_co_2.xml
.. _`Hanson et al. (2008)`: https://journals.ametsoc.org/view/journals/atot/26/8/2009jtecho650_1.xml
.. _`Portilla et al. (2009)`: https://journals.ametsoc.org/view/journals/atot/26/1/2008jtecho609_1.xml