.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

=======
Fitting
=======

Different functions are available for fitting parametric frequency spectra within the :py:mod:`~wavespectra.fit` subpackage:

* Pierson-Moskowitz
* Jonswap
* TMA



.. ipython:: python
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from wavespectra.fit.jonswap import jonswap
    from wavespectra.fit.pierson_moskowitz import pierson_moskowitz
    farr = np.arange(0.03, 0.3, 0.001)
    freq = xr.DataArray(
        farr,
        coords={"freq": farr},
        dims=("freq",),
        name="freq"
    )


Pierson-Moskowitz
-----------------
Pierson-Moskowitz spectrum (`Pierson and Moskowitz, 1964`_)).

.. ipython:: python
    :okexcept:
    :okwarning:

    dset = pierson_moskowitz(2, 10, freq)

    hs = float(dset.spec.hs())
    tp = float(dset.spec.tp())

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset.plot(label=f"Hs={hs:0.0f}m, Tp={tp:0.0f}s");

    @supress
    plt.legend();

    @savefig pm_1d.png
    plt.draw()

Jonswap
-------
Jonswap paremetric spectrum (`Hasselmann et al., 1973`_).

.. ipython:: python
    :okwarning:

    dset1 = jonswap(hs=2, tp=10, freq=freq, gamma=3.3)
    dset2 = jonswap(hs=2, tp=10, freq=freq, gamma=2.0)

    hs1 = float(dset1.spec.hs())
    tp1 = float(dset1.spec.tp())
    hs2 = float(dset2.spec.hs())
    tp2 = float(dset2.spec.tp())

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label=f"Gamma=3.3, Hs={hs1:0.0f}m, Tp={tp1:0.0f}s");
    dset2.plot(label=f"Gamma=2.0, Hs={hs2:0.0f}m, Tp={tp2:0.0f}s");

    @suppress
    plt.legend()

    @savefig jonswap_1d.png
    plt.draw()

When the peak enhancement `gamma` is 1 or less Jonswap becomes a Pierson-Moskowitz spectrum:

.. ipython:: python
    :okwarning:

    dset1 = pierson_moskowitz(hs=2, tp=10, freq=freq)
    dset2 = jonswap(hs=2, tp=10, freq=freq, gamma=1.0)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="Pierson-Moskowitz", linewidth=10);
    dset2.plot(label="Jonswap with gamma=1", linewidth=3);

    @suppress
    plt.legend()

    @savefig pm_jonswap_gamma1.png
    plt.draw()


TMA
---
TMA parametric spectrum (`Bouws et al., 1985`_).




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



.. _`Pierson and Moskowitz, 1964`: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JZ069i024p05181
.. _`Hasselmann et al., 1973`: https://www.researchgate.net/publication/256197895_Measurements_of_wind-wave_growth_and_swell_decay_during_the_Joint_North_Sea_Wave_Project_JONSWAP
.. _`Bouws et al., 1985`: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JC090iC01p00975
