.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

=======
Fitting
=======

Different functions are available for fitting parametric frequency spectra:


Pierson-Moskowitz
-----------------
Pierson-Moskowitz spectrum (`Pierson and Moskowitz, 1964`_)).


Jonswap
-------
Jonswap paremetric spectrum (`Hasselmann et al., 1973`_).


.. ipython:: python
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from wavespectra.fit.jonswap import jonswap
    farr = np.arange(0.03, 0.3, 0.001)
    freq = xr.DataArray(farr, coords={"freq": farr}, dims=("freq",), name="freq")
    dset1 = jonswap(hs=2, tp=10, freq=freq, gamma=3.3)
    dset2 = jonswap(hs=2, tp=10, freq=freq, gamma=1.0)

    hs1 = float(dset1.spec.hs())
    tp1 = float(dset1.spec.tp())
    hs2 = float(dset2.spec.hs())
    tp2 = float(dset2.spec.tp())

    @suppress
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    dset1.plot(ax=ax, label=f"Gamma=3.3, Hs={hs1:0.0f}m, Tp={tp1:0.0f}s");
    dset2.plot(ax=ax, label=f"Gamma=1.0, Hs={hs2:0.0f}m, Tp={tp2:0.0f}s");

    @suppress
    plt.legend()

    @savefig jonswap_1d.png
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
