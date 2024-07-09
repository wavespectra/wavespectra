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

Wavespectra works by defining some :doc:`conventions <conventions>` for spectral data
in xarray objects:

.. ipython:: python
    :okwarning:

    import numpy as np
    import xarray as xr
    import wavespectra
    # Spectral data in m2/Hz/degree
    efth = np.array([0.0000e+00, 1.0942e-03, 5.8909e-02, 2.0724e-01,
                     2.0267e-01, 1.6886e-01, 1.1980e-01, 7.7867e-02,
                     4.9664e-02, 3.2361e-02, 2.0902e-02, 1.3622e-02,
                     8.8645e-03, 5.6254e-03, 3.3722e-03, 2.0276e-03,
                     1.1350e-03, 4.5993e-04, 2.0441e-04, 0.0000e+00])
    # Frequency in Hz
    freq = np.array([0.05 , 0.075, 0.1  , 0.125, 0.15 , 0.175, 0.2  ,
                     0.225, 0.25 , 0.275, 0.3  , 0.325, 0.35 , 0.375,
                     0.4  , 0.425, 0.45 , 0.475, 0.5  , 0.525])
    # Direction in degree
    dir = np.array([0])
    # The DataArray and Dataset objects below are ready to use with wavespectra
    da = xr.DataArray(
        data=np.expand_dims(efth, 1),
        dims=["freq", "dir"],
        coords=dict(freq=freq, dir=dir),
        name="efth",
    )
    da.spec.hs()
    dset = da.to_dataset()
    dset.spec.hs()

The `spec` namespace
--------------------

Once wavespectra has been imported, `DataArray`_ and `Dataset`_ objects following this
convention will have the ``spec`` accessor available, from which the functionality of
wavespectra can be accessed.

SpecArray
~~~~~~~~~

The :py:class:`~wavespectra.specarray.SpecArray` accessor extends `DataArray`_ objects
and is the main entry point for spectral data manipulation methods.

.. ipython:: python
    :okwarning:

    dset.efth.spec

SpecDataset
~~~~~~~~~~~

The :py:class:`~wavespectra.specdataset.SpecDataset` accessor extends `Dataset`_ objects
to allow accessing :py:class:`~wavespectra.specarray.SpecArray` methods directly from
the Dataset, in addition to providing other methods such as writing spectral data files.

.. ipython:: python
    :okwarning:

    dset.spec

Reading spectra from files
--------------------------

Wavespectra provides functions to read various file formats including spectral wave
models like WAVEWATCHIII, SWAN and WWM, observation instruments such as TRIAXYS and
SPOTTER, and industry standard formats including ERA5, NDBC, Octopus among others.

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    from wavespectra import read_ww3

    dset = read_ww3("_static/ww3file.nc")
    dset

In version 4, xarray engines have been defined for all wavespectra readers, allowing
for direct reading of spectral data using `xarray.open_dataset`_.

.. ipython:: python
    :okwarning:

    dset = xr.open_dataset("_static/ww3file.nc", engine="ww3")


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


Interpolate
-----------
A custom interpolation method takes care of the cyclic nature of the wave direction.

.. ipython:: python
    :okwarning:

    ds = dset.efth.isel(site=0, time=0).sortby("dir")
    freq = np.arange(ds.freq.min(), ds.freq.max()+0.001, 0.001)
    dir = np.arange(0, 360, 1)
    ds_interp = ds.spec.interp(freq=freq, dir=dir)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ds.plot(ax=axs[0], x="dir", y="freq", cmap="turbo", add_colorbar=False)
    ds_interp.plot(ax=axs[1], x="dir", y="freq", cmap="turbo", add_colorbar=False)

    @savefig quickstart_interp.png
    plt.draw()


Smoothing
---------
Spectra smoothing is available using a running average method.

.. ipython:: python
    :okwarning:

    ds_smooth = ds.spec.smooth(freq_window=5, dir_window=5)
    dss = xr.concat([np.log10(ds), np.log10(ds_smooth)], dim="smooth")
    dss["smooth"] = ["false", "true"]
    dss.plot(col="smooth", x="dir", y="freq", cmap="turbo", add_colorbar=False);

    @savefig quickstart_smooth.png
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

Wavespectra wraps the plotting functionality from `xarray`_ to allow easy
plotting of frequency-direction spectral plots in polar coordinates.

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


Construction
------------

Spectral construction functionality has been implemented in version 4 with different
shape functions available for frequency and direction such as Jonswap and Cartwright:

.. ipython:: python
    :okwarning:

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
    p = plt.scatter(dset.lon, dset.lat, s=250, c=dset.isel(time=0).spec.hs(), cmap="turbo", marker="$\u25EF$", label="Dataset points");
    p = plt.scatter(idw.lon, idw.lat, s=80, c=idw.isel(time=0).spec.hs(), cmap="turbo", marker="o", edgecolor="k", label="Interpolated point");

    @suppress
    plt.legend(); plt.colorbar(p, label="Hs (m)")

    @savefig interp_stations_plot.png
    plt.draw()

The `nearest` neighbour and `bbox` options are also available besides inverse distance weighting (idw).



.. _SpecArray: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specarray.py
.. _SpecDataset: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specdataset.py
.. _xarray: https://xarray.pydata.org/en/stable/
.. _`xarray.open_dataset`: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
.. _xarray_plot: https://xarray.pydata.org/en/stable/plotting.html
.. _faceting: https://xarray.pydata.org/en/stable/plotting.html#faceting
.. _selecting: https://xarray.pydata.org/en/latest/indexing.html
.. _interpolating: https://xarray.pydata.org/en/latest/interpolation.html
.. _DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _`Hanson et al. (2008)`: https://journals.ametsoc.org/doi/pdf/10.1175/2009JTECHO650.1
.. _`WAVEWATCHIII`: https://github.com/NOAA-EMC/WW3
