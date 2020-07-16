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
    ds.spec.plot.contourf();

Parameters
----------

Frequency-direction spectra can be easily plotted in the period space.

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period.png
    ds.spec.plot.contourf(as_period=True);

By default the :math:`log10(efth)` is plotted but actual values can be shown instead.

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period_realvalues.png
    ds.spec.plot.contourf(as_period=True, as_log10=False, show_direction_label=True);

Plotting parameters from xarray can be prescribed.

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_xarray_parameters.png
    ds.spec.plot.contourf(
        cmap="viridis",
        vmin=-5,
        vmax=-2,
        levels=15,
        add_colorbar=False,
    );

Exclusive plotting parameters from wavespectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    * **as_log10** (True): Plot the log10 of the spectrum for better visualisation.
    * **as_period** (False): Plot spectra as period instead of frequency.
    * **show_radius_label** (True): Display the radius labels.
    * **show_direction_label** (False): Display the direction labels.

Default xarray parameters set by wavespectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    * **projection**: "polar"
    * **cmap**: `cmocean`_.cm.thermal

Radius extents
--------------

The radius extents can be controlled either by slicing / splitting frequencies or by setting axis properties.

Xarray's `selecting`_ methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_ax_extent1.png
    ds.sel(freq=slice(0.0, 0.2)).spec.plot.contourf(cmap="gray_r");

Wavespectra's :py:meth:`~wavespectra.specarray.SpecArray.split` method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_ax_extent2.png
    ds.spec.split(fmin=0, fmax=0.2).spec.plot.contourf(cmap="gray_r");

Matplotlib's axis properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okwarning:

    @suppress
    fig = plt.figure(figsize=figsize)

    ds.spec.plot.contourf(cmap="gray_r");
    ax = plt.gca()
    ax.set_rmin(0)
    ax.set_rmax(0.2)

    @savefig single_polar_plot_ax_extent3.png
    plt.draw()

Faceting
--------

Xarray's faceting capability is fully supported.

.. ipython:: python
    :okwarning:

    @savefig faceted_polar_plot2.png
    dset.isel(site=0).spec.plot.contourf(
        col="time",
        col_wrap=3,
        levels=15,
        figsize=(15,8),
        vmax=-1,
        cmap="jet"
    )

Setting clean axis is useful if plotting up many small axes for overview.

.. ipython:: python
    :okwarning:

    @savefig faceted_polar_plot3.png
    dset.isel(site=0).sel(freq=slice(0, 0.2)).spec.plot.contourf(
        col="time",
        col_wrap=3,
        levels=15,
        figsize=(15,8),
        vmax=-1,
        clean_radius=True,
        clean_sector=True
    )


Plotting types
--------------

Wavespectra supports xarray's `contour`_, `contourf`_ and `pcolormesh`_ plotting types. 

Contour
~~~~~~~
.. ipython:: python
    :okwarning:

    ds = dset.isel(site=0, time=range(2))
    @savefig contour_type_plot.png
    ds.spec.plot.contour(col="time");

Contourf
~~~~~~~~
.. ipython:: python
    :okwarning:

    @savefig contourf_type_plot.png
    ds.spec.plot.contourf(col="time");


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
