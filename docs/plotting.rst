.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

========
Plotting
========

Simplest usage
--------------

Wavespectra wraps the plotting functionality from `xarray`_ to allow easily defining
frequency-direction spectral plots in polar coordinates.

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    from wavespectra import read_swan

    @suppress
    figsize = (6, 4)

    dset = read_swan("_static/swanfile.spec", as_site=True)
    ds = dset.isel(site=0, time=0)

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot.png
    ds.spec.plot();


Parameters
----------

Frequency-direction spectra can be easily plotted in the period space.

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period.png
    ds.spec.plot(as_period=True);

By default the :math:`log10(efth * 1e3)` is plotted but actual values can be shown instead.

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period_realvalues.png
    ds.spec.plot(as_period=True, as_log10=False);

Plotting parameters from xarray can be prescribed.

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_xarray_parameters.png
    ds.spec.plot(
        cmap="turbo",
        vmin=0.005,
        vmax=1.0,
        add_colorbar=False,
    );

.. note::

    **Exclusive plotting parameters from wavespectra:**

    * **as_log10** (True): Plot the log10 of the spectrum for better visualisation.
    * **as_period** (False): Plot spectra as period instead of frequency.
    * **show_radii_labels** (True): Display the radii tick labels.
    * **show_theta_labels** (False): Display the directions tick labels.

    **Default xarray parameters set by  wavespectra:**

    * **projection**: "polar"
    * **cmap**: "RdBu_r".
    * **xlabel**, **ylabel**: Disabled.

Radius extents
--------------

The radius extents are controlled from `rmin` and `rmax` parameters.

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    ds.spec.plot(
        rmin=0,
        rmax=0.2,
        as_log10=False,
        normalised=False,
        levels=15,
        cmap="gray_r",
        radii_ticks=[0.05, 0.1, 0.15],
        radii_labels=["0.05", "0.1", "0.15Hz"],
        cbar_ticks=np.arange(0, 0.18, 0.02),
    );

    @savefig single_polar_plot_ax_extent3.png
    plt.draw()


Faceting
--------

Xarray's faceting capability is fully supported.

.. ipython:: python
    :okwarning:

    @savefig faceted_polar_plot2.png
    dset.isel(site=0, time=slice(None, 4)).spec.plot(
        col="time",
        col_wrap=2,
        figsize=(15,8),
        cmap="pink_r"
    )

Removing tick labels can be useful if plotting up many small axes for a more clear overview.

.. ipython:: python
    :okwarning:

    @savefig faceted_polar_plot3.png
    dset.isel(site=0).sel(freq=slice(0, 0.2)).spec.plot(
        col="time",
        col_wrap=3,
        figsize=(15,8),
        vmax=1,
        show_theta_labels=False,
        show_radii_labels=False
    )

Plotting types
--------------

Wavespectra supports xarray's `contour`_, `contourf`_ and `pcolormesh`_ plotting types. 

.. warning::

    contour broken for only one spectrum


Contour
~~~~~~~
.. ipython:: python
    :okwarning:

    ds = dset.isel(site=0, time=[0, 1])
    @savefig contour_type_plot.png
    ds.spec.plot(kind="contour", col="time", col_wrap=2);

Contourf
~~~~~~~~
.. ipython:: python
    :okwarning:

    @savefig contourf_type_plot.png
    ds.spec.plot(kind="contourf", col="time", col_wrap=1);

Pcolormesh
~~~~~~~~~~
.. ipython:: python
    :okwarning:

    @savefig pcolormesh_type_plot.png
    ds.spec.plot(kind="pcolormesh", col="time", col_wrap=2);


.. _SpecArray: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specarray.py
.. _SpecDataset: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/specdataset.py
.. _xarray: https://xarray.pydata.org/en/stable/
.. _selecting: https://xarray.pydata.org/en/latest/indexing.html
.. _xarray_plot: https://xarray.pydata.org/en/stable/plotting.html
.. _faceting: https://xarray.pydata.org/en/stable/plotting.html#faceting
.. _DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _contour: https://xarray.pydata.org/en/stable/generated/xarray.plot.contour.html#xarray.plot.contour
.. _contourf: https://xarray.pydata.org/en/stable/generated/xarray.plot.contourf.html#xarray.plot.contourf
.. _pcolormesh: https://xarray.pydata.org/en/stable/generated/xarray.plot.pcolormesh.html#xarray.plot.pcolormesh
.. _`Hanson et al. (2008)`: https://journals.ametsoc.org/doi/pdf/10.1175/2009JTECHO650.1
.. _cmocean: https://matplotlib.org/cmocean/
