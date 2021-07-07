Two-dimensional fit
___________________


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
    from wavespectra.directional import cartwright, bunney
    f = np.arange(0.03, 0.401, 0.001)
    d = np.arange(0, 360, 1)
    freq = xr.DataArray(f, {"freq": f}, ("freq",), "freq")
    dir = xr.DataArray(d, {"dir": d}, ("dir",), "dir")


Directional spreading functions
-------------------------------

Two directional spreading functions are currently implemented in wavespectra:

* :func:`~wavespectra.directional.cartwright`
* :func:`~wavespectra.directional.bunney` (TODO)


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

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dm = 60
    for dspr in [20, 30, 40, 50]:
        gth = cartwright(dir, dm, dspr=dspr)
        gth.plot(label=f"$D_m$={dm:0.0f} deg, $\sigma$={dspr} deg");

    @suppress
    plt.legend();

    @savefig cartwright_function.png
    plt.draw()


Bunney
~~~~~~

The Asymmetrical distribution of `Bunney et al. (2014)`_ addresses the skewed directional shape
under turning wind seas. The function modifies the peak direction and the directional
spread for each frequency above the peak so that

:math:`\frac{\displaystyle \partial{\theta}}{\displaystyle \partial{f}}=\frac{\displaystyle \theta_{p}-\theta_{m}}{\displaystyle f_{p}-f_{m}}`,

:math:`\frac{\displaystyle \partial{\sigma}}{\displaystyle \partial{f}}=\frac{\displaystyle \sigma_{p}-\sigma_{m}}{\displaystyle f_{p}-f_{m}}`

where 


Frequency-direction spectrum
----------------------------

Frequency-directional spectra :math:`E_{d}(f,d)` can be constructed from spectral wave parameters
by applying a directional spreading function to a parametric frequency spectrum:

.. ipython:: python
    :okexcept:
    :okwarning:

    ef = fit_jonswap(freq=freq, hs=2, tp=10, gamma=2.0)
    gth = cartwright(dir=dir, dm=135, dspr=25)
    efth = ef * gth

    @suppress
    fig = plt.figure(figsize=(6, 4))

    efth.spec.plot();

    @savefig jonswap_2d.png
    plt.draw()


Constructor function
~~~~~~~~~~~~~~~~~~~~

The Constructor :func:`~wavespectra.construct.construct_partition` defines an api to construct spectra
for a partition from available fit and spreading functions:

.. ipython:: python
    :okexcept:
    :okwarning:

    from wavespectra.construct import construct_partition

    efth = construct_partition(
        fit_name="fit_tma",
        fit_kwargs={"freq": freq, "hs": 2, "tp": 10, "dep": 10},
        dir_name="cartwright",
        dir_kwargs={"dir": dir, "dm": 225, "dspr": 15}
    )

    @suppress
    fig = plt.figure(figsize=(6, 4))

    efth.spec.plot(cmap="Spectral_r", add_colorbar=False);

    @savefig tma_2d.png
    plt.draw()


Fitting multiple spectra
------------------------

.. ipython:: python
    :okexcept:
    :okwarning:

    n = 9
    gamma = np.linspace(1, 3.3, n)
    hs = xr.DataArray(n*[2], {"gamma": gamma}, ("gamma",))
    tp = xr.DataArray(n*[10], {"gamma": gamma}, ("gamma",))
    dep = xr.DataArray(n*[30], {"gamma": gamma}, ("gamma",))

    efth = construct_partition(
        fit_name="fit_tma",
        fit_kwargs={"freq": freq, "hs": hs, "tp": tp, "dep": dep, "gamma": hs.gamma},
        dir_name="cartwright",
        dir_kwargs={"dir": dir, "dm": 225, "dspr": 20}
    )

    efth.spec.plot(
        normalised=False,
        as_period=False,
        logradius=False,
        levels=20,
        cmap="turbo",
        figsize=(12,12),
        show_theta_labels=False,
        radii_ticks=np.array([0.1, 0.15, 0.2]),
        rmin=0.05,
        rmax=0.22,
        add_colorbar=False,
        col="gamma",
        col_wrap=3,
    );

    @savefig parameter_checking.png
    plt.draw()


Comparing parametric spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okexcept:
    :okwarning:

    gamma = 3.3
    hs = 2
    tp = 10
    dep = 15
    dm = 225
    dspr = 20
    tm01 = 8
    tm02 = 7.5

    # Pierson-Moskowitz
    efth_pm = construct_partition(fit_name="fit_pierson_moskowitz", fit_kwargs={"freq": freq, "hs": hs, "tp": tp}, dir_name="cartwright", dir_kwargs={"dir": dir, "dm": dm, "dspr": dspr})
    # Jonswap
    efth_jswap = construct_partition(fit_name="fit_jonswap", fit_kwargs={"freq": freq, "hs": hs, "tp": tp, "gamma": gamma}, dir_name="cartwright", dir_kwargs={"dir": dir, "dm": dm, "dspr": dspr})
    # TMA
    efth_tma = construct_partition(fit_name="fit_tma", fit_kwargs={"freq": freq, "hs": hs, "tp": tp, "dep": dep, "gamma": gamma}, dir_name="cartwright", dir_kwargs={"dir": dir, "dm": dm, "dspr": dspr})
    # Gaussian
    efth_gaus = construct_partition(fit_name="fit_gaussian", fit_kwargs={"freq": freq, "hs": hs, "fp": 1/tp, "tm01": tm01, "tm02": tm02}, dir_name="cartwright", dir_kwargs={"dir": dir, "dm": dm, "dspr": dspr})
    # Concat along the new "method" dimension
    efth = xr.concat([efth_pm, efth_jswap, efth_tma, efth_gaus], dim="method")
    efth["method"] = ["Pierson-Moskowitz", "Jonswap", "TMA", "Gaussian"]

    efth.spec.plot(
        normalised=True,
        as_period=False,
        logradius=True,
        figsize=(8,8),
        show_theta_labels=False,
        add_colorbar=False,
        col="method",
        col_wrap=2,
    );

    @savefig compare_parametric_2d.png
    plt.draw()


.. include:: reconstruct.rst


.. _`Bunney et al. (2014)`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
.. _`Cartwright (1963)`: https://repository.tudelft.nl/islandora/object/uuid:b6c19f1e-cb31-4733-a4fb-0f685706269b
