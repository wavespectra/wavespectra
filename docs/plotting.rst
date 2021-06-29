.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

========
Plotting
========

Wavespectra wraps the plotting functionality from `xarray`_ to allow easily defining
frequency-direction spectral plots in polar coordinates.

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import read_swan
    dset = read_swan("_static/swanfile.spec", as_site=True)
    ds = dset.isel(site=0, time=0, drop=True)


Simplest usage
--------------

The :py:meth:`~wavespectra.SpecArray.plot` method is available in :py:class:`~wavespectra.SpecArray`. The simplest usage takes no arguments 
and attempts to define sensible settings for plotting normalised spectra on logarithmic scales:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    figsize = (6, 4)

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot.png
    ds.spec.plot();


Wave period spectrum
--------------------

Frequency-direction spectra can be easily plotted in the period space.

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period.png
    ds.spec.plot(as_period=True);

Normalised
----------

The spectrum is normalised by default as :math:`\frac{E_{d}(f,d)}{\max{E_{d}(f,d)}}` but the actual values can be shown instead:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period_realvalues.png
    ds.spec.plot(as_period=True, normalised=False, cmap="pink_r");


Logarithmic radii
-----------------

Radii are shown in a logarithmic scale by default. Linear radii can be defined by setting `logradius=False`. Default intervals 
for linear radii are defined for frequency and period types but they can be overwritten from the `radii_ticks` paramater:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period_linear_radii.png
    ds.spec.plot(
        as_period=True,
        normalised=False,
        cmap="pink_r",
        logradius=False,
        radii_ticks=[5, 10, 15, 20],
    );


.. note::

    The `as_log10` option to plot the :math:`\log{E_{d}(f,d)}` has been deprecated but similar result can be achieved 
    by calculating :math:`\log{E_{d}(f,d)}` beforehand:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    ds1 = ds.where(ds>0, 1e-5) # Avoid infinity values
    ds1 = np.log10(ds1)

    @savefig replicate_as_log10.png
    ds1.spec.plot(
        rmin=1,
        rmax=25,
        cmap=cmocean.cm.thermal_r,
        as_period=True,
        levels=20,
        cbar_ticks=[1, 3, 5, 7],
        cbar_kwargs={"label": "$\log{E_{d}(f,d)}$"},
        extend="both",
        efth_min=None
    );


Plotting parameters from xarray
-------------------------------

Wavespectra allows passing some parameters from the functions wrapped from xarray such as `contourf <http://xarray.pydata.org/en/stable/generated/xarray.plot.contourf.html>`_ 
(excluding some that are manipulated in wavespectra such as `ax`, `x` and others):

.. ipython:: python
    :okwarning:
    :okexcept:

    import matplotlib

    @savefig single_polar_plot_xarray_parameters.png
    ds.spec.plot(
        kind="contourf",
        as_period=True,
        normalised=False,
        cmap="turbo",
        add_colorbar=False,
        extend="both",
        levels=25,
    );

.. warning::

    **Some of the xarray parameters that are not exposed in wavespectra:**

    * **projection**: Always set to "polar".
    * **x**, **y**: Set to wavespectra coordinates naming.
    * **xlabel**, **ylabel**: Disabled.
    * **ax**, **aspect**, **size**: Conflict with axes defined in wavespectra.
    * **xlim**, **ylim**: produce no effect.

Radii extent
------------

The radii extent are controlled from `rmin` and `rmax` parameters.

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    ds.spec.plot(
        rmin=0,
        rmax=0.15,
        logradius=False,
        normalised=False,
        levels=25,
        cmap="gray_r",
        radii_ticks=[0.03, 0.06, 0.09, 0.12, 0.15],
        radii_labels=["0.05", "0.1", "0.15Hz"],
        cbar_ticks=np.arange(0, 0.18, 0.02),
    );

    @savefig single_polar_plot_ax_extent3.png
    plt.draw()


.. note::

    **Exclusive plotting parameters from wavespectra:**

    * **kind** ("contourf") : Plot kind, one of ("contourf", "contour", "pcolormesh").
    * **normalised** (True): Show :math:`E(f,d)` normalised between 0 and 1.
    * **logradius** (True): Set log radii.
    * **as_period** (False): Set radii as wave period instead of frequency.
    * **show_radii_labels** (True): Display the radii tick labels.
    * **show_theta_labels** (False): Display the directions tick labels.
    * **cbar_ticks** ([1e-2, 1e-1, 1e0]): Tick values for colorbar.

Faceting
--------

Xarray's faceting capability is fully supported.

.. ipython:: python
    :okwarning:
    :okexcept:

    @savefig faceted_polar_plot2.png
    dset.isel(site=0, time=slice(None, 4)).spec.plot(
        col="time",
        col_wrap=2,
        figsize=(15,8),
        cmap="Spectral_r"
    )

Clean axes
----------

Removing tick labels can be useful if plotting up many small axes for a more clear overview.

.. ipython:: python
    :okwarning:
    :okexcept:

    @savefig faceted_polar_plot3.png
    dset.isel(site=0).sel(freq=slice(0, 0.2)).spec.plot(
        col="time",
        col_wrap=3,
        figsize=(15,8),
        vmax=1,
        show_theta_labels=False,
        show_radii_labels=False
    )

    @suppress
    plt.close("all")


Plotting types
--------------

Wavespectra supports xarray's `contour`_, `contourf`_ and `pcolormesh`_ plotting types. 

Contour
~~~~~~~
.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig contour_type_plot.png
    ds.spec.plot(kind="contour");

Contourf
~~~~~~~~
.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig contourf_type_plot.png
    ds.spec.plot(kind="contourf");

Pcolormesh
~~~~~~~~~~
.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig pcolormesh_type_plot.png
    ds.spec.plot(
        kind="pcolormesh",
        cbar_ticks=np.arange(0, 1.1, 0.1),
        vmin=0,
        vmax=1.0,
        cmap="gray_r",
    );


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
