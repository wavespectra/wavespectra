.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

===========
Quick start
===========

Wavespectra is an open source project for processing ocean wave spectral data.
The library is built on top of xarray and provides reading and writing of different
spectral data formats, calculation of common integrated wave paramaters, spectral
partitioning and spectral manipulation in a package focussed on speed and efficiency
for large numbers of spectra.

Reading spectra from files
--------------------------

Several methods are provided to read various file formats including spectral wave
models like WAVEWATCHIII, SWAN and WWM, observation instruments such as TRIAXYS and
SPOTTER, and industry standard formats including ERA5, NDBC, Octopus among others.

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    from wavespectra import read_ww3

    dset = read_ww3("_static/ww3file.nc")
    dset

The `spec` namespace
--------------------

Wavespectra defines a new namespace accessor called `spec` which is attached to
`xarray`_ objects. This namespace provides access to several methods from the two main
objects in wavespectra:

* :py:class:`~wavespectra.specarray.SpecArray`
* :py:class:`~wavespectra.specdataset.SpecDataset`

which extend functionality from xarray's `DataArray`_ and `Dataset`_ respectively.

SpecArray
~~~~~~~~~

.. ipython:: python
    :okwarning:

    dset.efth.spec

SpecDataset
~~~~~~~~~~~

.. ipython:: python
    :okwarning:

    dset.spec

Spectral wave parameters
------------------------
Several methods are available to calculate integrated wave parameters. They can be
accessed from both SpecArray (`efth` variable) and SpecDataset accessors:

.. ipython:: python
    :okwarning:

    hs = dset.efth.spec.hs()
    hs
    hs1 = dset.spec.hs()
    hs.identical(hs1)

    @suppress
    plt.figure(figsize=(8,5))

    hs.plot.line(x="time");

    @suppress
    plt.legend(("Site 1", "Site 2"))
    @suppress
    plt.title("")

    @savefig hs.png
    plt.draw()


.. ipython:: python
    :okwarning:

    stats = dset.spec.stats(
        ["hs", "hmax", "tp", "tm01", "tm02", "dpm", "dm", "dspr", "swe"]
    )
    stats

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(8, 6))

    stats.hs.plot.line(ax=ax1, x="time");
    @suppress
    ax1.set_ylabel("$Hs$ (m)")

    stats.hmax.plot.line(ax=ax2, x="time");
    @suppress
    ax2.set_ylabel("$Hmax$ (m)")

    stats.dpm.plot.line(ax=ax3, x="time");
    @suppress
    ax3.set_ylabel("$Dpm$ (deg)")

    stats.dspr.plot.line(ax=ax4, x="time");
    @suppress
    ax4.set_ylabel("$Dspr$ (deg)")

    stats.tp.plot.line(ax=ax5, x="time");
    @suppress
    ax5.set_ylabel("$Tp$ (s)")

    stats.tm01.plot.line(ax=ax6, x="time");
    @suppress
    ax6.set_ylabel("$Tm01$ (s)")

    @suppress
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]: ax.set_xlabel(""); ax.tick_params(bottom=False, labelbottom=False); ax.get_legend().remove()    

    @savefig many_stats.png
    plt.draw()


Spectra file writing
--------------------
Several methods are available in the `SpecDataset` accessor for writing spectral data to
different file formats. The following example writes the dataset to a SWAN ASCII file:

.. ipython:: python
    :okwarning:

    dset.spec.to_swan("specfile.swn")

    !head -n 40 specfile.swn


Plotting
--------

Wavespectra wraps the plotting functionality from `xarray`_ to allow easily defining
frequency-direction spectral plots in polar coordinates.

.. ipython:: python
    :okwarning:

    ds = dset.isel(site=0, time=[0, 1]).spec.split(fmin=0.05, fmax=0.4)
    @savefig faceted_polar_plot.png
    ds.spec.plot(
        kind="contourf",
        col="time",
        as_period=False,
        normalised=True,
        logradius=True,
        add_colorbar=False,
        figsize=(8, 5)
    );


Plotting Hovmoller diagrams of frequency spectra timeseries can be done in only a few lines.

.. ipython:: python
    :okwarning:

    import cmocean

    @suppress
    plt.figure(figsize=(8, 4))

    ds = dset.isel(site=0).spec.split(fmax=0.18).spec.oned().rename({"freq": "period"})
    ds = ds.assign_coords({"period": 1 / ds.period})
    ds.period.attrs.update({"standard_name": "sea_surface_wave_period", "units": "s"})

    @savefig hovmoller_plot.png
    ds.plot.contourf(x="time", y="period", vmax=1.25, cmap=cmocean.cm.thermal, levels=10);


Partitioning
------------

Different partitioning techniques are available within the `spec.partition` namespace.
The partitioning methods follow the naming convention defined in the `WAVEWATCHIII`_
model (`ptm1`, `ptm2`, etc) with the addition of some custom methods. In the following
example, the `ptm1` method is used to partition the dataset into wind sea and three
swells (`ptm1` is equivalent to the former `spec.partition()` method deprecated in 
version 4).

.. ipython:: python
    :okwarning:

    dspart = dset.spec.partition.ptm1(dset.wspd, dset.wdir, dset.dpt)
    pstats = dspart.spec.stats(["hs", "dpm"])
    pstats

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    hs.isel(site=0).plot(ax=ax1, label='Full spectrum', marker='o');
    pstats.hs.isel(part=0, site=0).plot(ax=ax1, label='Partition 0 (sea)', marker='o');
    pstats.hs.isel(part=1, site=0).plot(ax=ax1, label='Partition 1 (swell 1)', marker='o');
    pstats.hs.isel(part=2, site=0).plot(ax=ax1, label='Partition 2 (swell 2)', marker='o');
    pstats.hs.isel(part=3, site=0).plot(ax=ax1, label='Partition 3 (swell 3)', marker='o');

    @suppress
    plt.legend(loc=0, fontsize=8); ax1.set_title(""); ax1.set_ylabel("$Hs$ (m)"); ax1.set_xlabel(""); ax1.set_xticklabels([])

    dset.spec.dpm().isel(site=0).plot(ax=ax2, label='Full spectrum', marker='o');
    pstats.dpm.isel(part=0, site=0).plot(ax=ax2, label='Partition 0 (sea)', marker='o');
    pstats.dpm.isel(part=1, site=0).plot(ax=ax2, label='Partition 1 (swell 1)', marker='o');
    pstats.dpm.isel(part=2, site=0).plot(ax=ax2, label='Partition 2 (swell 2)', marker='o');
    pstats.dpm.isel(part=3, site=0).plot(ax=ax2, label='Partition 3 (swell 3)', marker='o');

    @suppress
    plt.legend(loc=0, fontsize=8); ax2.set_title(""); ax2.set_ylabel("$Dpm$ (deg)"); ax2.set_xlabel("")

    @savefig watershed_hs.png
    plt.draw()


Reconstruction
--------------

Spectral reconstruction functionality has been implemented in version 4 with different
shape functions available for frequency and direction such as Jonswap and Cartwright:

.. ipython:: python
    :okwarning:

    import numpy as np
    from wavespectra.construct import construct_partition
    freq = np.arange(0.03, 0.401, 0.001)
    dir = np.arange(0, 360, 1)
    ds = construct_partition(
        freq_name="jonswap",
        dir_name="cartwright",
        freq_kwargs={"freq":  freq, "fp": 0.1, "gamma": 3.3, "hs": 1.5},
        dir_kwargs={"dir": dir, "dm": 60, "dspr": 30},
    )
    @savefig reconstruted_polar.png
    ds.spec.plot();

.. ipython:: python
    :okwarning:

    @savefig reconstruted_1d.png
    ds.spec.oned().plot(figsize=(8, 4));

Selecting
---------

Wavespectra complements xarray's selecting_ and interpolating_ functionality with functions to select and
interpolate from `site` coordinates with the :py:meth:`~wavespectra.specdataset.SpecDataset.sel` method.

.. ipython:: python
    :okwarning:

    idw = dset.spec.sel(
        lons=[92, 92.05, 92.1, 92.1, 92.1, 92.1, 92.05, 92, 92, 92],
        lats=[19.8, 19.8, 19.8, 19.85, 19.9, 19.95, 19.95, 19.95, 19.9, 19.85],
        method="idw"
    )
    idw

    @suppress
    plt.figure(figsize=(8, 4.5))
    p = plt.scatter(dset.lon, dset.lat, 200, dset.isel(time=0).spec.hs(), cmap="turbo", marker="v", edgecolor="k", label="Dataset points");
    p = plt.scatter(idw.lon, idw.lat, 80, idw.isel(time=0).spec.hs(), cmap="turbo", marker="o", edgecolor="k", label="Interpolated point");

    @suppress
    plt.legend(); plt.colorbar(p, label="Hs (m)")

    @savefig interp_stations_plot.png
    plt.draw()

The `nearest` neighbour and `bbox` options are also available besides inverse distance weighting (idw).



.. _SpecArray: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specarray.py
.. _SpecDataset: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specdataset.py
.. _xarray: https://xarray.pydata.org/en/stable/
.. _xarray_plot: https://xarray.pydata.org/en/stable/plotting.html
.. _faceting: https://xarray.pydata.org/en/stable/plotting.html#faceting
.. _selecting: https://xarray.pydata.org/en/latest/indexing.html
.. _interpolating: https://xarray.pydata.org/en/latest/interpolation.html
.. _DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _`Hanson et al. (2008)`: https://journals.ametsoc.org/doi/pdf/10.1175/2009JTECHO650.1
.. _`WAVEWATCHIII`: https://github.com/NOAA-EMC/WW3