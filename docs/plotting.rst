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
    :okexcept:

    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import read_swan, read_era5
    dset = read_era5("_static/era5file.nc").isel(time=0)
    ds = dset.isel(lat=0, lon=0)


Simplest usage
--------------

The :py:meth:`~wavespectra.SpecArray.plot` method is available in :py:class:`~wavespectra.SpecArray`. The simplest usage takes no arguments 
and defines sensible settings for plotting normalised spectra on logarithmic radii and countour levels:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    figsize = (6, 4)

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot.png
    ds.spec.plot();


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
    ds.spec.plot(kind="contour", colors="#af1607", linewidths=0.5);

Contourf
~~~~~~~~
.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig contourf_type_plot.png
    ds.spec.plot(kind="contourf", cmap=cmocean.cm.thermal);

Pcolormesh
~~~~~~~~~~
.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig pcolormesh_type_plot.png
    ds.spec.plot(kind="pcolormesh", cmap=cmocean.cm.thermal);


Wave period spectrum
--------------------

Frequency-direction spectra can be easily plotted in the period space.

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period.png
    ds.spec.plot(as_period=True, cmap="pink_r");

Normalised
----------

The normalised spectrum :math:`\frac{E_{d}(f,d)}{\max{E_{d}}}` is plotted by default but the actual values can be shown instead:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period_realvalues.png
    ds.spec.plot(as_period=True, normalised=False, cmap="Spectral_r");

Logarithmic contour levels are only default for normalised spectra but they can be still manually specified:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period_realvalues_loglevels.png
    ds.spec.plot(
        as_period=True,
        normalised=False,
        cmap="Spectral_r",
        levels=np.logspace(np.log10(0.005), np.log10(0.4), 15),
        cbar_ticks=[0.01, 0.1, 1],
    );


Logarithmic radii
-----------------

Radii are shown in a logarithmic scale by default. Linear radii can be defined by setting `logradius=False` 
(radii ticks can be prescribed from the `radii_ticks` paramater):

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    @savefig single_polar_plot_period_linear_radii.png
    ds.spec.plot(
        as_period=True,
        normalised=False,
        levels=15,
        cmap="bone_r",
        logradius=False,
        radii_ticks=[5, 10, 15, 20, 25],
    );


.. hint::

    The `as_log10` option to plot the :math:`\log{E_{d}(f,d)}` has been deprecated but similar result 
    can be achieved by calculating the :math:`\log{E_{d}(f,d)}` beforehand:

.. ipython:: python
    :okwarning:
    :okexcept:

    @suppress
    fig = plt.figure(figsize=figsize)

    ds1 = ds.where(ds>0, 1e-5) # Avoid infinity values
    ds1 = np.log10(ds1)

    @savefig replicate_as_log10.png
    ds1.spec.plot(
        as_period=True,
        logradius=False,
        cbar_kwargs={"label": "Normalised $\log{E_{d}(f,d)}$"},
        vmin=0.39,
        levels=15,
        extend="both",
        cmap=cmocean.cm.thermal,
    );


Radii extents
-------------

The radii extents are controlled from `rmin` and `rmax` parameters:

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
        radii_labels_angle=120,
        radii_labels_size=7,
    );

    @savefig single_polar_plot_ax_extent3.png
    plt.draw()


.. admonition:: Exclusive plotting parameters from wavespectra

    * **kind** ("contourf") : Plot kind, one of ("contourf", "contour", "pcolormesh").
    * **normalised** (True): Plot the normalised :math:`E(f,d)` between 0 and 1.
    * **logradius** (True): Set log radii.
    * **as_period** (False): Set wave period radii instead of frequency.
    * **show_radii_labels** (True): Display the radii tick labels.
    * **show_theta_labels** (False): Display the directions tick labels.
    * **radii_ticks** (array): Tick values for radii.
    * **radii_labels_angle** (22.5): Polar angle at which radii labels are positioned.
    * **radii_labels_size** (8): Fontsize for radii labels.
    * **cbar_ticks**: Tick values for colorbar (default depends if normalised, logradius and as_period).
    * **clean_axis** (False): Remove radii and theta ticks for a clean view.


Plotting parameters from xarray
-------------------------------

Wavespectra allows passing some parameters from the functions wrapped from xarray such as `contourf <http://xarray.pydata.org/en/stable/generated/xarray.plot.contourf.html>`_ 
(excluding some that are manipulated in wavespectra such as `ax`, `x` and others):

.. ipython:: python
    :okwarning:
    :okexcept:

    @savefig single_polar_plot_xarray_parameters.png
    ds.spec.plot(
        kind="contourf",
        cmap="turbo",
        add_colorbar=False,
        extend="both",
        levels=25,
    );

.. admonition:: Some of the xarray parameters that are not exposed in wavespectra
    :class: warning

    * **projection**: Always set to "polar".
    * **x**, **y**: Set to wavespectra coordinates naming.
    * **xlabel**, **ylabel**: Disabled.
    * **ax**, **aspect**, **size**: Conflict with axes defined in wavespectra.
    * **xlim**, **ylim**: produce no effect.


Faceting
--------

Xarray's faceting capability is fully supported.

.. ipython:: python
    :okwarning:
    :okexcept:

    dset.spec.plot(
        col="lon",
        row="lat",
        figsize=(16,8),
        add_colorbar=False,
        show_theta_labels=False,
        show_radii_labels=True,
        radii_ticks=[0.05, 0.1, 0.2, 0.4],
        rmax=0.4,
        radii_labels_size=5,
        cmap="Spectral_r",
    );
    plt.tight_layout()
    @savefig faceted.png
    plt.draw()

Clean axes
----------

Use the `clean_axis` argument to remove radii and theta grids for a clean overview. This is
equivalent to disabling ticks from the axis by calling `ax.set_rticks=[]`, `ax.set_xticks=[]`.

.. ipython:: python
    :okwarning:
    :okexcept:

    dset1 = dset.where(dset>0, 1e-5)
    dset1 = np.log10(dset1)

    dset1.spec.plot(
        clean_axis=True,
        col="lon",
        row="lat",
        figsize=(16,8),
        logradius=False,
        vmin=0.39,
        levels=15,
        extend="both",
        cmap=cmocean.cm.thermal,
        add_colorbar=False,
    );
    plt.tight_layout()
    @savefig faceted_cleanaxis.png
    plt.draw()

    @suppress
    plt.close("all")


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
