.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

===================
Two-dimensional fit
===================

Frequency-direction spectra can be fit by applying a directional spreding function to a
frequency spectrum.


.. ipython:: python
    :okexcept:
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import fit_pierson_moskowitz, fit_jonswap, fit_tma, fit_gaussian
    from wavespectra.directional_distribution import cartwright, bunney
    f = np.arange(0.03, 0.3, 0.001)
    d = np.arange(0, 360, 1)
    freq = xr.DataArray(f, {"freq": f}, ("freq",), "freq")
    dir = xr.DataArray(d, {"dir": d}, ("dir",), "dir")


Directional spreading functions
-------------------------------

Two directional spreading functions are currently implemented in wavespectra:

* :func:`~wavespectra.directional_distribution.cartwright`
* :func:`~wavespectra.directional_distribution.bunney` (TODO)


Cartwright
~~~~~~~~~~

The cosine-squared distribution of `Cartwright (1963)`_ assumes single mean direction and directional spread
for all frequencies with a symmetrical decay of energy around the peak represented by a cosine-squared function:

:math:`G(\theta,f)=F(s)cos^{2}\frac{1}{2}(\theta-\theta_{m})`

where :math:`\theta` is the wave direction, :math:`f` is the wave frequency, :math:`\theta_{m}` is the
mean direction and :math:`F(s)` is a scaling parameter.

.. ipython:: python
    :okexcept:
    :okwarning:

    dm = 60
    gth1 = cartwright(dir, dm, dspr=20)
    gth2 = cartwright(dir, dm, dspr=30)
    gth3 = cartwright(dir, dm, dspr=40)
    gth4 = cartwright(dir, dm, dspr=50)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    gth1.plot(label=f"$D_m$={dm:0.0f} deg, $\sigma$=20 deg");
    gth2.plot(label=f"$D_m$={dm:0.0f} deg, $\sigma$=30 deg");
    gth3.plot(label=f"$D_m$={dm:0.0f} deg, $\sigma$=40 deg");
    gth4.plot(label=f"$D_m$={dm:0.0f} deg, $\sigma$=50 deg");

    @suppress
    plt.legend();

    @suppress
    plt.grid(False)

    @savefig cartwright_function.png
    plt.draw()


.. attention::

    Do we want that the function dropps off to zero after +-90 degrees from the peak?


Bunney
~~~~~~

The Asymmetrical distribution of `Bunney et al. (2014)`_ addresses the skewed directional shape
often observed under turning wind seas. The function modifies the peak direction and the directional
spread for each frequency above the peak so that

:math:`\frac{\displaystyle \partial{\theta}}{\displaystyle \partial{f}}=\frac{\displaystyle \theta_{p}-\theta_{m}}{\displaystyle f_{p}-f_{m}}`,

:math:`\frac{\displaystyle \partial{\sigma}}{\displaystyle \partial{f}}=\frac{\displaystyle \sigma_{p}-\sigma_{m}}{\displaystyle f_{p}-f_{m}}`

where 

.. attention::

    * :math:`\theta_{p}` is the peak wave direction (`pdp0` - `sea_surface_wave_peak_from_direction_partition_0`)
    * :math:`\theta_{m}` is the mean wave direction (`pdir0` - `sea_surface_wave_from_direction_partition_0`)
    * :math:`f_{m}` is the mean wave frequency?
        * `pt01c0` - `sea_surface_wave_mean_period_t01_partition_0`?
        * `pt02c0` - `sea_surface_wave_mean_period_t02_partition_0`?
        * `ptm10c0` - `sea_surface_wave_mean_period_tm10_partition_0`?
    * :math:`f_{p}` is the peak wave frequency (`ptp0` - `sea_surface_wave_period_at_variance_spectral_density_maximum_partition_0`)
    * :math:`\sigma_{p}` is the peak spread (`psw0` - `sea_surface_wave_spectral_width_partition_0`)?
    * :math:`\sigma_{m}` is the mean spread (`psw0` - `sea_surface_wave_spectral_width_partition_0`)?

    Only for partitions actively driven by wind?


.. _`Bunney et al. (2014)`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
.. _`Cartwright (1963)`: https://repository.tudelft.nl/islandora/object/uuid:b6c19f1e-cb31-4733-a4fb-0f685706269b