Directional spreading
_____________________


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
    from wavespectra.construct.frequency import pierson_moskowitz, jonswap, tma, gaussian
    @suppress
    from wavespectra.construct.direction import cartwright, asymmetric
    @suppress
    from wavespectra.construct import construct_partition
    @suppress
    freq = np.arange(0.03, 0.401, 0.001)
    @suppress
    dir = np.arange(0, 360, 1)


Two directional spreading functions are currently implemented in wavespectra:

* Symmetrical cosine-squared distribution of Cartwright (1963): :func:`~wavespectra.construct.direction.cartwright`
* Asymmetrical directional distribution of Bunney et al. (2014): :func:`~wavespectra.construct.direction.asymmetric`


Cartwright symmetrical spread
-----------------------------

The cosine-squared distribution of `Cartwright (1963)`_ assumes single mean direction
and directional spread for all frequencies with a symmetrical decay of energy around
the peak represented by a cosine-squared function:

:math:`G(\theta,f)=F(s)cos^{2}\frac{1}{2}(\theta-\theta_{m})`

where :math:`\theta` is the wave direction, :math:`f` is the wave frequency,
:math:`\theta_{m}` is the mean direction and :math:`F(s)` is a scaling parameter.

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


.. tip::

    Relevant wavespectra stats methods for the :meth:`~wavespectra.construct.direction.cartwright` function:

    * Mean wave direction :meth:`~wavespectra.SpecArray.dm`.
    * Mean direction spread :meth:`~wavespectra.SpecArray.dspr`.


Bunney asymmetrical spread
--------------------------

The Asymmetrical distribution of `Bunney et al. (2014)`_ addresses the skewed
directional shape under turning wind seas. The function modifies the peak direction and
the directional spread for each frequency above the peak so that

:math:`\frac{\displaystyle \partial{\theta}}{\displaystyle \partial{f}}=\frac{\displaystyle \theta_{p}-\theta_{m}}{\displaystyle f_{p}-f_{m}}`,

:math:`\frac{\displaystyle \partial{\sigma}}{\displaystyle \partial{f}}=\frac{\displaystyle \sigma_{p}-\sigma_{m}}{\displaystyle f_{p}-f_{m}}`

where :math:`\theta` is the wave direction, :math:`\sigma` is the directional spread,
:math:`f` is the wave frequency and subscripts :math:`p` and :math:`m` denote peak and
mean respectively. The gradients are used to modify the wave direction and directional
spread :math:`\forall f \geq f_p`:

:math:`\theta=\theta_p + \frac{\displaystyle \partial{\theta}}{\displaystyle \partial{f}} (f-f_p),`

:math:`\sigma=\sigma_p + \frac{\displaystyle \partial{\sigma}}{\displaystyle \partial{f}} (f-f_p)`

with :math:`\theta=\theta_p` and :math:`\sigma=\sigma_p` :math:`\forall f<f_p`. These
equations imply :math:`f=f_p \Rightarrow \theta=\theta_p` and
:math:`f=f_p \Rightarrow \sigma=\sigma_p` with values linearly increasing or decreasing
above the frequency peak at rates defined by the gradients
:math:`\frac{\partial{\theta}}{\partial{f}}` and :math:`\frac{\partial{\sigma}}{\partial{f}}`.

The asymmetrical distribution is designed for wind sea systems satisfying the following
constraints:

:math:`T_m<10s`, and

:math:`rt < 0.9`.

.. ipython:: python
    :okexcept:
    :okwarning:

    asym = asymmetric(
        dir=dir,
        freq=freq,
        dm=45,
        dpm=50,
        dspr=20,
        dpspr=17,
        fm=0.1,
        fp=0.09,
    )

    @savefig asymmetric_distribution.png
    asym.spec.plot();


.. tip::

    Relevant wavespectra stats methods for the :meth:`~wavespectra.construct.direction.asymmetric` function:

    * Mean wave direction :meth:`~wavespectra.SpecArray.dm`.
    * Peak wave direction :meth:`~wavespectra.SpecArray.dpm` (or alternatively :meth:`~wavespectra.SpecArray.dp`).
    * Mean direction spread :meth:`~wavespectra.SpecArray.dspr`.
    * Peak direction spread :meth:`~wavespectra.SpecArray.dpspr`.
    * Mean wave period :meth:`~wavespectra.SpecArray.tm01` (or alternatively :meth:`~wavespectra.SpecArray.tm02`).
    * Peak wave frequency :meth:`~wavespectra.SpecArray.fp`.


Parameter sensitivity
~~~~~~~~~~~~~~~~~~~~~

The gradients :math:`\frac{\partial{\theta}}{\partial{f}}` and
:math:`\frac{\partial{\sigma}}{\partial{f}}` define the shape of the asymmetrical
spread function of `Bunney et al. (2014)`_. Here we briefly examine sensitivity to
:math:`\theta`, :math:`\sigma` and :math:`f` parameters:


Sensitivity to :math:`\partial{\theta}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ipython:: python
    :okexcept:
    :okwarning:

    dim = "$d_p-d_m$"
    kw["dm"] = xr.DataArray(
        [49., 45., 40.],
        coords={dim: [49, 45, 40]},
        dims=(dim,),
    )
    kw["dpm"] = xr.full_like(kw["dm"], 50)
    kw["dspr"] = xr.full_like(kw["dm"], 20)
    kw["dpspr"] = xr.full_like(kw["dm"], 17)
    kw["fm"] = xr.full_like(kw["dm"], 0.1)
    kw["fp"] = xr.full_like(kw["dm"], 0.09)

    asym = asymmetric(dir=dir, freq=freq, **kw)
    # Just for titles
    asym = asym.assign_coords({dim: (kw["dpm"] - kw["dm"]).values})

    @savefig bunney_distribution_vary_dm.png
    asym.spec.plot(col=dim, add_colorbar=False, figsize=(12, 5), logradius=False);


Sensitivity to :math:`\partial{\sigma}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ipython:: python
    :okexcept:
    :okwarning:

    dim = "$\sigma_p-\sigma_m$"
    kw["dspr"] = xr.DataArray(
        [19.9, 19.5, 19.1],
        coords={dim: [19.9, 19.5, 19.1]},
        dims=(dim,),
    )
    kw["dpspr"] = xr.full_like(kw["dspr"], 20)
    kw["dpm"] = xr.full_like(kw["dspr"], 50)
    kw["dm"] = xr.full_like(kw["dspr"], 45)
    kw["fm"] = xr.full_like(kw["dspr"], 0.1)
    kw["fp"] = xr.full_like(kw["dspr"], 0.09)

    asym = asymmetric(dir=dir, freq=freq, **kw)
    asym = asym.assign_coords({dim: (kw["dpspr"] - kw["dspr"]).values})

    @savefig bunney_distribution_vary_dspr.png
    asym.spec.plot(col=dim, add_colorbar=False, figsize=(12, 5), logradius=False);

Sensitivity to :math:`\partial{f}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ipython:: python
    :okexcept:
    :okwarning:

    dim = "$f_p-f_m$"
    kw = {}
    kw["fm"] = xr.DataArray(
        [0.095, 0.1, 0.15],
        coords={dim: [0.095, 0.1, 0.15]},
        dims=(dim,),
    )
    kw["fp"] = xr.full_like(kw["fm"], 0.09)
    kw["dspr"] = xr.full_like(kw["fm"], 19.5)
    kw["dpspr"] = xr.full_like(kw["fm"], 20)
    kw["dpm"] = xr.full_like(kw["fm"], 50)
    kw["dm"] = xr.full_like(kw["fm"], 45)

    asym = asymmetric(dir=dir, freq=freq, **kw)
    asym = asym.assign_coords({dim: (kw["fp"] - kw["fm"]).values})

    @savefig bunney_distribution_vary_fm.png
    asym.spec.plot(col=dim, add_colorbar=False, figsize=(12, 5), logradius=False);


Frequency-direction spectrum
____________________________

Frequency-directional spectra :math:`E_{d}(f,d)` can be constructed from spectral wave
parameters by applying a directional spreading function to a parametric frequency
spectrum:

.. ipython:: python
    :okexcept:
    :okwarning:

    ef = jonswap(freq=freq, fp=0.1, gamma=2.0, hs=2.0)
    gth = cartwright(dir=dir, dm=135, dspr=25)
    efth = ef * gth

    @suppress
    fig = plt.figure(figsize=(6, 4))

    efth.spec.plot();

    @savefig jonswap_2d.png
    plt.draw()


Symmetrical vs asymmetrical
---------------------------

Two-dimensional spectra can be constructed from existing frequency spectra. This
example constructs symmetrical and asymmetrical directional distributions to
one-dimensional spectrum :math:`E_d(f)` integrated from existing :math:`E_d(f,d)`.

.. ipython:: python
    :okexcept:
    :okwarning:

    dset = read_ww3("_static/ww3file.nc")
    dset = dset.spec.partition.ptm1(
        dset.wspd,
        dset.wdir,
        dset.dpt,
    )
    dset = dset.isel(time=0, site=-1, part=0, drop=True).sortby("dir")

    ds = dset.spec.stats(["dm", "dpm", "dspr", "dpspr", "tm01", "fp"])
    ds["fm"] = 1 / ds.tm01

    # Define directional distributions
    c = cartwright(dir=dset.dir, dm=ds.dm, dspr=ds.dspr)
    a = asymmetric(
        dir=dset.dir,
        freq=dset.freq,
        dm=ds.dm,
        dpm=ds.dpm,
        dspr=ds.dspr,
        dpspr=ds.dpspr,
        fm=ds.fm,
        fp=ds.fp,
    )

    # Apply directional distributions to the one-dimensional spectrum
    dscart = dset.spec.oned() * c
    dsasym = dset.spec.oned() * a

    dsall = xr.concat([dset, dscart, dsasym], dim="fit")
    dsall["fit"] = ["Original", "Cartwright", "Asymmetric"]
    dsall.spec.plot(
        figsize=(12, 5),
        col="fit",
        logradius=False,
        rmax=0.5,
        add_colorbar=False,
        show_theta_labels=False,
    );

    @savefig original_cartwright_asymmetric.png
    plt.draw()


Constructor function
--------------------

The :func:`~wavespectra.construct.construct_partition` constructor defines an api to
construct two-dimensional spectra for a partition from frequency shape and directional
spread functions available in wavespectra:

.. ipython:: python
    :okexcept:
    :okwarning:

    efth = construct_partition(
        freq_name="tma",
        freq_kwargs={"freq": freq, "fp": 0.1, "dep": 10, "hs": 2.0},
        dir_name="cartwright",
        dir_kwargs={"dir": dir, "dm": 225, "dspr": 15}
    )

    @suppress
    fig = plt.figure(figsize=(6, 4))

    efth.spec.plot(cmap="Spectral_r", add_colorbar=False);

    @savefig tma_2d.png
    plt.draw()


Here we use the constructor to define 2D spectra with four different spectral shapes: 

.. ipython:: python
    :okexcept:
    :okwarning:

    gamma = 3.3
    hs = 2
    tp = 10
    fp = 1 / tp
    dep = 15
    dm = 225
    dspr = 20
    gw = 0.07
    d_kw = dict(dir_name="cartwright", dir_kwargs=dict(dir=dir, dm=dm, dspr=dspr))
    # Pierson-Moskowitz
    f_kw = dict(freq_name="pierson_moskowitz", freq_kwargs=dict(freq=freq, hs=hs, fp=fp))
    efth_pm = construct_partition(**{**f_kw, **d_kw})
    # Jonswap
    f_kw = dict(freq_name="jonswap", freq_kwargs=dict(freq=freq, hs=hs, fp=fp, gamma=gamma))
    efth_jswap = construct_partition(**{**f_kw, **d_kw})
    # TMA
    f_kw = dict(freq_name="tma", freq_kwargs=dict(freq=freq, hs=hs, fp=fp, gamma=gamma, dep=dep))
    efth_tma = construct_partition(**{**f_kw, **d_kw})
    # Gaussian
    f_kw = dict(freq_name="gaussian", freq_kwargs=dict(freq=freq, hs=hs, fp=fp, gw=gw))
    efth_gaus = construct_partition(**{**f_kw, **d_kw})
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

In this example the constructor is used to construct multiple spectra with common
:func:`~wavespectra.construct.direction.cartwright` distribution and
:func:`~wavespectra.construct.frequency.jonswap` shape but varying the :math:`\gamma`
parameter:

.. ipython:: python
    :okexcept:
    :okwarning:

    n = 9
    gamma = np.linspace(1, 3.3, n)
    hs = xr.DataArray(n*[2], {"gamma": gamma}, ("gamma",))
    fp = xr.DataArray(n*[0.1], {"gamma": gamma}, ("gamma",))
    dep = xr.DataArray(n*[30], {"gamma": gamma}, ("gamma",))

    efth = construct_partition(
        freq_name="tma",
        freq_kwargs={"freq": freq, "fp": fp, "dep": dep, "gamma": hs.gamma, "hs": hs},
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


.. include:: reconstruction.rst


.. _`Bunney et al. (2014)`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
.. _`Cartwright (1963)`: https://repository.tudelft.nl/islandora/object/uuid:b6c19f1e-cb31-4733-a4fb-0f685706269b
