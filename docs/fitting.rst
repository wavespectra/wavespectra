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

Different functions are available for fitting parametric frequency spectra within the :py:mod:`~wavespectra.fit` subpackage:

* :func:`~wavespectra.fit.pierson_moskowitz.pierson_moskowitz`
* :func:`~wavespectra.fit.jonswap.jonswap`
* :func:`~wavespectra.fit.tma.tma`
* :func:`~wavespectra.fit.gaussian.gaussian`



.. ipython:: python
    :okexcept:
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from wavespectra.fit.pierson_moskowitz import pierson_moskowitz
    from wavespectra.fit.jonswap import jonswap
    from wavespectra.fit.tma import tma
    from wavespectra.fit.gaussian import gaussian
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

    dset = pierson_moskowitz(freq=freq, hs=2, tp=10)

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

    dset1 = jonswap(freq=freq, hs=2, tp=10, gamma=3.3)
    dset2 = jonswap(freq=freq, hs=2, tp=10, gamma=2.0)

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

    dset1 = pierson_moskowitz(freq=freq, hs=2, tp=10)
    dset2 = jonswap(freq=freq, hs=2, tp=10, gamma=1.0)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="Pierson-Moskowitz", linewidth=10);
    dset2.plot(label="Jonswap with $\gamma=1$", linewidth=3);

    @suppress
    plt.legend()

    @savefig pm_jonswap_gamma1.png
    plt.draw()


TMA
---
TMA spectral form for seas in water of finite depth (`Bouws et al., 1985`_).

.. ipython:: python
    :okexcept:
    :okwarning:

    dset1 = tma(freq=freq, hs=2, tp=10, dep=10)
    dset2 = tma(freq=freq, hs=2, tp=10, dep=50)

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

    dset1 = jonswap(freq=freq, hs=2, tp=10)
    dset2 = tma(freq=freq, hs=2, tp=10, dep=80)

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

    dset1 = gaussian(freq=freq, hs=2, fp=1/10, tm01=8, tm02=8)
    dset2 = gaussian(freq=freq, hs=2, fp=1/10, tm01=8, tm02=6)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    t0 = 1 / float(freq[0])
    dset1.plot(label=f"rt={(8-t0)/(8-t0):0.2f}");
    dset2.plot(label=f"rt={(8-t0)/(6.5-t0):0.2f}");

    @suppress
    plt.legend();

    @savefig gaussian_1d.png
    plt.draw()


Multiple fitting
----------------
When arguments to the function are DataArray objects, multiple spectra are fit
along each coordinate.

.. ipython:: python

    from wavespectra import read_swan
    from wavespectra.fit.jonswap import jonswap
    dset = read_swan("_static/swanfile.spec")
    hs = dset.spec.hs()
    tp = dset.spec.tp()
    
    ds = jonswap(
        hs=dset.spec.hs(),
        tp=dset.spec.tp(),
        freq=dset.freq,
        gamma=1.6
    )
    ds

    ds_ori = dset.spec.oned().isel(lat=0, lon=0, time=0, drop=True)
    ds_new = ds.isel(lat=0, lon=0, time=0, drop=True)

    @suppress
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ds_ori.plot(ax=ax, label="Original spectrum");
    ds_new.plot(ax=ax, label="Jonswap fitting");

    @suppress
    plt.legend()

    @savefig jonswap_original_fitting.png
    plt.draw()


~~~~~~~~~~~~~~~~~~~~~~~~
Directional distribution
~~~~~~~~~~~~~~~~~~~~~~~~

Cartwright
----------
Cosine-squared distribution of `Cartwright (1963)`_.


.. _`Pierson and Moskowitz, 1964`: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JZ069i024p05181
.. _`Hasselmann et al., 1973`: https://www.researchgate.net/publication/256197895_Measurements_of_wind-wave_growth_and_swell_decay_during_the_Joint_North_Sea_Wave_Project_JONSWAP
.. _`Bouws et al., 1985`: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JC090iC01p00975
.. _`Bunney et al., 2014`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
.. _`Cartwright (1963)`: https://repository.tudelft.nl/islandora/object/uuid:b6c19f1e-cb31-4733-a4fb-0f685706269b