.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

============
Partitioning
============

.. ipython:: python
    :okexcept:
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import read_ww3, read_swan
    from wavespectra import fit_pierson_moskowitz, fit_jonswap, fit_tma, fit_gaussian
    from wavespectra.directional import cartwright, bunney
    from wavespectra.construct import construct_partition, partition_and_reconstruct
    freq = np.arange(0.03, 0.401, 0.001)
    dir = np.arange(0, 360, 1)


One-dimensional fit
___________________

Different functions are available for fitting parametric spectral shapes.
The functions are defined within the :py:mod:`~wavespectra.fit` subpackage:

* :func:`~wavespectra.fit_pierson_moskowitz`
* :func:`~wavespectra.fit_jonswap`
* :func:`~wavespectra.fit_tma`
* :func:`~wavespectra.fit_gaussian`


Pierson-Moskowitz
-----------------

Pierson-Moskowitz spectral form for fully developed seas (`Pierson and Moskowitz, 1964`_):

:math:`S(f)=Af^{-5} \exp{(-Bf^{-4})}`.

.. ipython:: python
    :okexcept:
    :okwarning:

    dset = fit_pierson_moskowitz(freq=freq, hs=2, fp=0.1)

    hs = float(dset.spec.hs())
    tp = float(dset.spec.tp())

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset.plot(label=f"Hs={hs:0.0f}m, Tp={tp:0.0f}s");

    @suppress
    plt.legend();

    @savefig pm_1d.png
    plt.draw()


.. tip::

    Relevant wavespectra stats methods for the :func:`~wavespectra.fit_pierson_moskowitz` function:

    * Significant wave height :meth:`~wavespectra.SpecArray.hs`.
    * Peak wave period :meth:`~wavespectra.SpecArray.tp`.


Jonswap
-------

Jonswap spectral form for developing seas (`Hasselmann et al., 1973`_):

:math:`S(f) = \alpha g^2 (2\pi)^{-4} f^{-5} \exp{\left [-\frac{5}{4} \left (\frac{f}{f_p} \right)^{-4} \right]} \gamma^{\exp{[\frac{(f-f_p)^2}{2\sigma^2f_p^2}}]}`.

.. ipython:: python
    :okwarning:

    dset1 = fit_jonswap(freq=freq, fp=0.1, gamma=3.3, hs=2.0)
    dset2 = fit_jonswap(freq=freq, fp=0.1, gamma=2.0, hs=2.0)

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

    dset1 = fit_pierson_moskowitz(freq=freq, hs=2, fp=0.1)
    dset2 = fit_jonswap(freq=freq, hs=2, fp=0.1, gamma=1.0)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="Pierson-Moskowitz", linewidth=10);
    dset2.plot(label="Jonswap with $\gamma=1$", linewidth=3);

    @suppress
    plt.legend()

    @savefig pm_jonswap_gamma1.png
    plt.draw()

Compare against real frequency spectrum (with gamma adjusted for a good fit):

.. ipython:: python

    ds = read_swan("_static/swanfile.spec").isel(time=0, lat=0, lon=0, drop=True)
    ds_fit = fit_jonswap(
        freq=ds.freq,
        fp=ds.spec.fp(),
        gamma=1.6,
        hs=ds.spec.hs(),
    )

    @suppress
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ds.spec.oned().plot(ax=ax, label="Original spectrum");
    ds_fit.plot(ax=ax, label="Jonswap fitting");

    @suppress
    plt.legend()

    @savefig jonswap_original_fitting.png
    plt.draw()

The spectrum is scalled by :math:`\alpha` but if :math:`Hs` is provided it is used so that :math:`4\sqrt{m_0} = Hs`.

.. ipython:: python

    ds1 = fit_jonswap(freq=ds.freq, fp=ds.spec.fp(), gamma=ds.spec.gamma(), alpha=ds.spec.alpha())
    ds2 = fit_jonswap(freq=ds.freq, fp=ds.spec.fp(), gamma=ds.spec.gamma(), alpha=ds.spec.alpha(), hs=ds.spec.hs())

    @suppress
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ds.spec.oned().plot(ax=ax, label=f"Original spectrum ($Hs={float(ds.spec.hs()):0.2f}m$)");
    ds1.plot(ax=ax, label=f"Jonswap scaled by $\\alpha$ ($Hs={float(ds1.spec.hs()):0.2f}m$)");
    ds2.plot(ax=ax, label=f"Jonswap scaled by $Hs$ ($Hs={float(ds2.spec.hs()):0.2f}m$)");

    @suppress
    plt.legend(fontsize=9)

    ax.set_xlim([0, 0.3])

    @savefig jonswap_original_alpha_hs_scaled.png
    plt.draw()

.. tip::

    Relevant wavespectra stats methods for the :func:`~wavespectra.fit_jonswap` function:

    * Peak wave period :meth:`~wavespectra.SpecArray.fp`.
    * Peak enhancement factor :meth:`~wavespectra.SpecArray.gamma`.
    * Fetch dependant scaling coefficient :meth:`~wavespectra.SpecArray.alpha`.
    * Significant wave height :meth:`~wavespectra.SpecArray.hs`.


TMA
---

TMA spectral form for seas in water of finite depth (`Bouws et al., 1985`_):

:math:`S(f) = S_{J}(f) \tanh{kh}^2 (1 + \frac{2kh} {\sinh{2kh}})^{-1}`

.. ipython:: python
    :okexcept:
    :okwarning:

    dset1 = fit_tma(freq=freq, fp=0.1, dep=10, hs=2)
    dset2 = fit_tma(freq=freq, fp=0.1, dep=50, hs=2)

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

    dset1 = fit_jonswap(freq=freq, fp=0.1, hs=2.0)
    dset2 = fit_tma(freq=freq, fp=0.1, dep=80, hs=2.0)

    @suppress
    fig = plt.figure(figsize=(6, 4))

    dset1.plot(label="Jonswap", linewidth=10);
    dset2.plot(label="TMA in deep water", linewidth=3);

    @suppress
    plt.legend()

    @savefig jonswap_tma_deepwater.png
    plt.draw()


.. tip::

    Relevant wavespectra stats methods for the :func:`~wavespectra.fit_tma` function:

    * Peak wave frequency :meth:`~wavespectra.SpecArray.fp`.
    * Peak enhancement factor :meth:`~wavespectra.SpecArray.gamma`.
    * Fetch dependant scaling coefficient :meth:`~wavespectra.SpecArray.alpha`.
    * Significant wave height :meth:`~wavespectra.SpecArray.hs`.


Gaussian
--------

Gaussian spectral form for swell (`Bunney et al., 2014`_):

:math:`S(f)=\frac{\displaystyle m_0^2}{\displaystyle \sigma \sqrt{2\pi}} \exp{\left(-\frac{\displaystyle (f-f_p)^2}{\displaystyle 2\sigma^2}\right)}`

where :math:`m_0=\left(\frac{Hs}{4} \right)^2`, and the gaussian width :math:`\sigma` (:meth:`~wavespectra.SpecArray.gw`) is calculatd from the mean
:math:`T_m` (:meth:`~wavespectra.SpecArray.tm01`) and the zero-upcrossing :math:`T_z` (:meth:`~wavespectra.SpecArray.tm02`) as

:math:`\sigma=\sqrt{\frac{\displaystyle m_0}{\displaystyle T_z^2} - \frac{\displaystyle m_0^2}{\displaystyle T_m^2}}`.

The authors define a criterion for fitting a swell partition with the Gaussian
distribution based on the ratio :math:`rt` between :math:`T_m` and :math:`T_z`:

:math:`rt = \frac{(T_m - T_0)}{(T_z - T_0)} >= 0.95`

where :math:`T_0` is the period corresponding to the lowest frequency bin.

.. ipython:: python
    :okexcept:
    :okwarning:

    def sigma(hs, tm, tz):
        m0 = (hs / 4) ** 2
        return np.sqrt((m0 / (tz ** 2)) - (m0 ** 2 / tm ** 2))

    dset1 = fit_gaussian(freq=freq, hs=2, fp=1/10, gw=sigma(hs=2, tm=8.0, tz=6.5))
    dset2 = fit_gaussian(freq=freq, hs=2, fp=1/10, gw=sigma(hs=2, tm=8.0, tz=8.0))

    @suppress
    fig = plt.figure(figsize=(6, 4))

    t0 = 1 / float(freq[0])
    dset1.plot(label=f"rt={(8-t0)/(6.5-t0):0.2f}");
    dset2.plot(label=f"rt={(8-t0)/(8-t0):0.2f}");

    @suppress
    plt.legend();

    @savefig gaussian_1d.png
    plt.draw()


.. tip::

    Relevant wavespectra stats methods for the :func:`~wavespectra.fit_gaussian` function:

    * Significant wave height :meth:`~wavespectra.SpecArray.hs`.
    * Peak wave period :meth:`~wavespectra.SpecArray.tp`.
    * Gaussian width :meth:`~wavespectra.SpecArray.gw`.


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
    dspart_param = dspart.spec.stats(["hs", "fp", "gamma"])
    dspart_param["dpt"] = dset.dpt.expand_dims({"part": dspart.part})

    dspart_param


Spectra are fit along all coodinates in the DataArrays


.. ipython:: python
    :okexcept:
    :okwarning:

    dspart_jonswap = fit_jonswap(
        freq=dspart.freq,
        fp=dspart_param.fp,
        gamma=dspart_param.gamma,
        hs=dspart_param.hs,
    )
    dspart_tma = fit_tma(
        freq=dspart.freq,
        fp=dspart_param.fp,
        gamma=dspart_param.gamma,
        dep=dspart_param.dpt,
        hs=dspart_param.hs,
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


.. include:: fitting_2d.rst


.. _`Pierson and Moskowitz, 1964`: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JZ069i024p05181
.. _`Hasselmann et al., 1973`: https://www.researchgate.net/publication/256197895_Measurements_of_wind-wave_growth_and_swell_decay_during_the_Joint_North_Sea_Wave_Project_JONSWAP
.. _`Bouws et al., 1985`: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JC090iC01p00975
.. _`Bunney et al., 2014`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
.. _`Cartwright (1963)`: https://repository.tudelft.nl/islandora/object/uuid:b6c19f1e-cb31-4733-a4fb-0f685706269b
