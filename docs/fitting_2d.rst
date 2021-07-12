Two-dimensional fit
___________________


Frequency-direction spectra can be fit by applying a directional spreding function to a
frequency spectrum.


.. ipython:: python
    :okexcept:
    :okwarning:

    @suppress
    import numpy as np
    @suppress
    import xarray as xr
    @suppress
    import matplotlib.pyplot as plt
    @suppress
    import cmocean
    @suppress
    from wavespectra import read_ww3
    @suppress
    from wavespectra import fit_pierson_moskowitz, fit_jonswap, fit_tma, fit_gaussian
    @suppress
    from wavespectra.directional import cartwright, bunney
    @suppress
    from wavespectra.construct import construct_partition
    @suppress
    f = np.arange(0.03, 0.401, 0.001)
    @suppress
    d = np.arange(0, 360, 1)
    @suppress
    freq = xr.DataArray(f, {"freq": f}, ("freq",), "freq")
    @suppress
    dir = xr.DataArray(d, {"dir": d}, ("dir",), "dir")


Directional spreading functions
-------------------------------

Two directional spreading functions are currently implemented in wavespectra:

* :func:`~wavespectra.directional.cartwright`
* :func:`~wavespectra.directional.bunney`


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

where :math:`\theta` is the wave direction, :math:`\sigma` is the directional spread,
:math:`f` is the wave frequency and subscripts :math:`p` and :math:`m` denote
peak and mean respectively.

The gradients are used to modify the wave direction :math:`\theta` and directional spread :math:`\sigma_p`
above the frequency spectral peak :math:`f_p`:

:math:`\theta=\theta_p \frac{\displaystyle \partial{\theta}}{\displaystyle \partial{f}} (f-f_p), \forall f \geq f_p`

:math:`\theta=\theta_p, \forall f < f_p,`

and

:math:`\sigma=\sigma_p \frac{\displaystyle \partial{\sigma}}{\displaystyle \partial{f}} (f-f_p), \forall f \geq f_p`

:math:`\sigma=\sigma_p, \forall f < f_p`.

As defined, these equations would result in :math:`f=f_p \Rightarrow \theta=0, \sigma=0` which do not make physical sense.
We have implemented a slightly modified version of the method as described in `Bunney et al. (2014)`_ for :math:`f \geq f_p`:

:math:`\theta=\theta_p+\frac{\displaystyle \partial{\theta}}{\displaystyle \partial{f}} (f-f_p), \forall f \geq f_p`,

:math:`\sigma=\sigma_p+\frac{\displaystyle \partial{\sigma}}{\displaystyle \partial{f}} (f-f_p), \forall f \geq f_p`


.. ipython:: python
    :okexcept:
    :okwarning:

    b = bunney(dir=dir, freq=freq, dm=45, dpm=50, dspr=20, dpspr=17, fm=0.1, fp=0.09)

    @savefig bunney_distribution.png
    b.spec.plot()


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


Symmetrical vs asymmetrical spreading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython:: python
    :okexcept:
    :okwarning:

    dset = read_ww3("_static/ww3file.nc").isel(time=0, site=-1, drop=True).sortby("dir")
    dset = dset.spec.partition(dset.wspd, dset.wdir, dset.dpt).isel(part=0, drop=True)

    ds = dset.spec.stats(["dm", "dpm", "dspr", "dpspr", "tm01", "fp"])
    ds["fm"] = 1 / ds.tm01

    c = cartwright(dir=dset.dir, dm=ds.dm, dspr=ds.dspr)

    b = bunney(dir=dset.dir, freq=dset.freq, dm=ds.dm, dpm=ds.dpm, dspr=ds.dspr, dpspr=ds.dpspr, fm=ds.fm, fp=ds.fp)

    # Apply directional distributions to the one-dimensional spectrum
    dscart = dset.spec.oned() * c
    dsbun = dset.spec.oned() * b

    dsall = xr.concat([dset, dscart, dsbun], dim="fit")
    dsall["fit"] = ["Original", "Cartwright", "Bunney"]
    dsall.spec.plot(
        figsize=(12, 5),
        col="fit",
        logradius=False,
        rmax=0.5,
        add_colorbar=False,
        show_theta_labels=False,
    )

    @savefig original_cartwright_bunney.png
    plt.draw()


Constructor function
~~~~~~~~~~~~~~~~~~~~

The Constructor :func:`~wavespectra.construct.construct_partition` defines an api to construct spectra
for a partition from available fit and spreading functions:

.. ipython:: python
    :okexcept:
    :okwarning:

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
