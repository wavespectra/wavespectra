.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

==============
Reconstruction
==============

spectrum reconstruction from partitioned wave parameters following the method
of `Bunney et al., 2014`_.


~~~~~~~~~~~~~
Spectral form
~~~~~~~~~~~~~

Different functions are available for fitting parametric frequency spectral shapes. The functions are defined within the :py:mod:`~wavespectra.fit` subpackage:

* :func:`~wavespectra.fit_pierson_moskowitz`
* :func:`~wavespectra.fit_jonswap`
* :func:`~wavespectra.fit_tma`
* :func:`~wavespectra.fit_gaussian`



.. ipython:: python
    :okexcept:
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import fit_pierson_moskowitz, fit_jonswap, fit_tma, fit_gaussian
    farr = np.arange(0.03, 0.3, 0.001)
    freq = xr.DataArray(
        farr,
        coords={"freq": farr},
        dims=("freq",),
        name="freq"
    )


Pierson-Moskowitz
-----------------
Pierson-Moskowitz spectral form for fully developed seas (`Pierson and Moskowitz, 1964`_).

.. ipython:: python
    :okexcept:
    :okwarning:

    dset = fit_pierson_moskowitz(freq=freq, hs=2, tp=10)

    hs = float(dset.spec.hs())
    tp = float(dset.spec.tp())

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset.plot(label=f"Hs={hs:0.0f}m, Tp={tp:0.0f}s");

    @suppress
    plt.legend();

    @savefig pm_1d.png
    plt.draw()


Jonswap
-------
Jonswap spectral form for developing seas (`Hasselmann et al., 1973`_).

.. ipython:: python
    :okwarning:

    dset1 = fit_jonswap(freq=freq, hs=2, tp=10, gamma=3.3)
    dset2 = fit_jonswap(freq=freq, hs=2, tp=10, gamma=2.0)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="$\gamma=3.3$");
    dset2.plot(label="$\gamma=2.0$");

    @suppress
    plt.legend()

    @savefig jonswap_1d.png
    plt.draw()

When the peak enhancement :math:`\gamma=1` Jonswap becomes a Pierson-Moskowitz spectrum:

.. ipython:: python
    :okwarning:

    dset1 = fit_pierson_moskowitz(freq=freq, hs=2, tp=10)
    dset2 = fit_jonswap(freq=freq, hs=2, tp=10, gamma=1.0)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="Pierson-Moskowitz", linewidth=10);
    dset2.plot(label="Jonswap with $\gamma=1$", linewidth=3);

    @suppress
    plt.legend()

    @savefig pm_jonswap_gamma1.png
    plt.draw()

Compare against real frequency spectrum with gamma adjusted for a good fit:

.. ipython:: python

    from wavespectra import read_swan
    ds = read_swan("_static/swanfile.spec").isel(time=0, lat=0, lon=0, drop=True)
    ds_fit = fit_jonswap(
        freq=ds.freq,
        hs=ds.spec.hs(),
        tp=ds.spec.tp(),
        gamma=1.6,
    )

    @suppress
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ds.spec.oned().plot(ax=ax, label="Original spectrum");
    ds_fit.plot(ax=ax, label="Jonswap fitting");

    @suppress
    plt.legend()

    @savefig jonswap_original_fitting.png
    plt.draw()


TMA
---
TMA spectral form for seas in water of finite depth (`Bouws et al., 1985`_).

.. ipython:: python
    :okexcept:
    :okwarning:

    dset1 = fit_tma(freq=freq, hs=2, tp=10, dep=10)
    dset2 = fit_tma(freq=freq, hs=2, tp=10, dep=50)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="Depth=10");
    dset2.plot(label="Depth=50");

    @suppress
    plt.legend();

    @savefig tma_1d.png
    plt.draw()

In deep water TMA becomes a Jonswap spectrum:

.. ipython:: python
    :okexcept:
    :okwarning:

    dset1 = fit_jonswap(freq=freq, hs=2, tp=10)
    dset2 = fit_tma(freq=freq, hs=2, tp=10, dep=80)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="Jonswap", linewidth=10);
    dset2.plot(label="TMA in deep water", linewidth=3);

    @suppress
    plt.legend()

    @savefig jonswap_tma_deepwater.png
    plt.draw()


Gaussian
--------
Gaussian spectral form for swell (`Bunney et al., 2014`_). The authors define a criterion for choosing the gaussian fit based on the ratio :math:`rt` between the mean :math:`T_m` (:meth:`~wavespectra.SpecArray.tm01`) and the zero-upcrossing :math:`T_z` (:meth:`~wavespectra.SpecArray.tm02`) spectral periods:

:math:`rt = \frac{(T_m - T_0)}{(T_z - T_0)} >= 0.95`

where :math:`T_0` is the period corresponding to the lowest frequency bin.

.. ipython:: python
    :okexcept:
    :okwarning:

    dset1 = fit_gaussian(freq=freq, hs=2, fp=1/10, tm01=8, tm02=8)
    dset2 = fit_gaussian(freq=freq, hs=2, fp=1/10, tm01=8, tm02=6)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    t0 = 1 / float(freq[0])
    dset1.plot(label=f"rt={(8-t0)/(8-t0):0.2f}");
    dset2.plot(label=f"rt={(8-t0)/(6.5-t0):0.2f}");

    @suppress
    plt.legend();

    @savefig gaussian_1d.png
    plt.draw()


Fitting multiple spectra
------------------------

Fitting function parameters can be DataArrays with multiple dimensions
such as times and watershed partitions:

.. ipython:: python
    :okexcept:
    :okwarning:

    from wavespectra import read_wwm
    dset = read_wwm("_static/wwmfile.nc").isel(site=0, drop=True)

    dspart = dset.spec.partition(dset.wspd, dset.wdir, dset.dpt)
    dspart_param = dspart.spec.stats(["hs", "tp", "gamma"])
    dspart_param["dpt"] = dset.dpt.expand_dims({"part": dspart.part})

    dspart_param


Spectra are fit along all coodinates in the DataArray


.. ipython:: python
    :okexcept:
    :okwarning:

    dspart_jonswap = fit_jonswap(
        freq=dspart.freq,
        hs=dspart_param.hs,
        tp=dspart_param.tp,
        gamma=dspart_param.gamma,
    )
    dspart_tma = fit_tma(
        freq=dspart.freq,
        hs=dspart_param.hs,
        tp=dspart_param.tp,
        gamma=dspart_param.gamma,
        dep=dspart_param.dpt,
    )
    dspart_tma


Compare fits for the first swell partition:

.. ipython:: python
    :okexcept:
    :okwarning:

    cmap = cmocean.cm.thermal
    fig = plt.figure(figsize=(12, 10))

    # Original spectra
    ax = fig.add_subplot(311)
    ds = dspart.spec.oned().isel(part=1).transpose("freq", "time")
    ds.plot.contourf(cmap=cmap, levels=20, ylim=(0.02, 0.4), vmax=4.0);

    @suppress
    ax.set_title("Original spectra")
    @suppress
    ax.set_xticklabels([])
    @suppress
    ax.set_xlabel("")

    # Jonswap fit
    ax = fig.add_subplot(312)
    ds = dspart_jonswap.isel(part=1).transpose("freq", "time")
    ds.plot.contourf(cmap=cmap, levels=20, ylim=(0.02, 0.4), vmax=4.0);

    @suppress
    ax.set_title("Jonswap fit")
    @suppress
    ax.set_xticklabels([])
    @suppress
    ax.set_xlabel("")

    # TMA fit
    ax = fig.add_subplot(313)
    ds = dspart_tma.isel(part=1).transpose("freq", "time")
    ds.plot.contourf(cmap=cmap, levels=20, ylim=(0.02, 0.4), vmax=4.0);

    @suppress
    ax.set_title("TMA fit")

    @savefig frequency_spectra_timeseries_original_fits.png
    plt.draw()


~~~~~~~~~~~~~~~~~~~~~~~~
Directional distribution
~~~~~~~~~~~~~~~~~~~~~~~~

Cartwright
----------
Cosine-squared distribution of `Cartwright (1963)`_.


Bunney
------
Swell Gaussian distribution of `Bunney et al., (2014)`_.

TODO


.. _`Pierson and Moskowitz, 1964`: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JZ069i024p05181
.. _`Hasselmann et al., 1973`: https://www.researchgate.net/publication/256197895_Measurements_of_wind-wave_growth_and_swell_decay_during_the_Joint_North_Sea_Wave_Project_JONSWAP
.. _`Bouws et al., 1985`: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JC090iC01p00975
.. _`Bunney et al., 2014`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
.. _`Cartwright (1963)`: https://repository.tudelft.nl/islandora/object/uuid:b6c19f1e-cb31-4733-a4fb-0f685706269b