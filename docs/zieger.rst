Zieger approach
----------------

Zieger defined three spectra reconstruction options based on Cartwright spread and Jonswap fits.
The methods differ in how they specify some Jonswap parameters.

.. admonition:: Method 1
    :class: note

    Default Jonswap parameters.

    * Default :math:`\gamma=3.3`.

    * Default :math:`\sigma_a=0.7`.

    * Default :math:`\sigma_b=0.9`.

    * :math:`\alpha=\frac{5\pi^4}{g^2}Hs^2f_{p}^{4}`

.. admonition:: Method 2
    :class: note

    Gaussian width :math:`g_w` used to define the widths :math:`\sigma_a`, :math:`\sigma_b` of the peak enhancement factor :math:`\gamma`.

    * :math:`\gamma` calculated from the spectra.

    * :math:`\sigma_a=g_w` (but capped at min=0.04, max=0.09).

    * :math:`\sigma_b=g_w+0.1`.

    * :math:`\alpha=\frac{5\pi^4}{g^2}Hs^2f_{p}^{4}`


.. admonition:: Method 3
    :class: note

    Scale :math:`Hs` for very small partitions.

    * Bump :math:`Hs` by 12% to calculate :math:`\alpha` if :math:`Hs<0.7m`.

    * Otherwise same as method 2.


Below are examples on how to implement the methods defined from Zieger from wavespectra.

First define some input data:

.. ipython:: python
    :okexcept:
    :okwarning:

    import numpy as np
    import xarray as xr
    from wavespectra import read_ww3

    # Reading and partitioning existing spectrum
    dset = read_ww3("_static/ww3file.nc").isel(time=0, site=-1, drop=True).sortby("dir")
    dsetp = dset.spec.partition.ptm1(dset.wspd, dset.wdir, dset.dpt)

    # Calculating parameters
    ds = dsetp.spec.stats(["fp", "dm", "dspr", "gamma", "gw", "hs"])

    # Alpha
    ds["alpha"] = (5 * np.pi**4 / 9.81**2) * ds.hs**2 * ds.fp**4

    # Alpha for method #3
    hs = ds.hs.where(ds.hs >= 0.7, ds.hs * 1.12)
    ds["alpha3"] = (5 * np.pi**4 / 9.81**2) * hs**2 * ds.fp**4

    # Common reconstruct parameters
    dir_name = "cartwright"
    dir_kwargs = dict(dir=dset.dir, dm=ds.dm, dspr=ds.dspr)
    freq_name = "jonswap"
    kw = dict(freq=dset.freq, fp=ds.fp)


Reconstruct from method 1

.. ipython:: python
    :okexcept:
    :okwarning:

    freq_kwargs = {**kw, **dict(gamma=3.3, sigma_a=0.7, sigma_b=0.9, alpha=ds.alpha)}
    method1 = construct_partition(freq_name, dir_name, freq_kwargs, dir_kwargs)
    method1 = method1.max(dim="part")


Reconstruct from method 2

.. ipython:: python
    :okexcept:
    :okwarning:

    sa = ds.gw.where(ds.gw >= 0.04, 0.04).where(ds.gw <= 0.09, 0.09)
    sb = sa + 0.1
    freq_kwargs = {**kw, **dict(gamma=ds.gamma, sigma_a=sa, sigma_b=sb, alpha=ds.alpha)}
    method2 = construct_partition(freq_name, dir_name, freq_kwargs, dir_kwargs)
    method2 = method2.max(dim="part")


Reconstruct from method 3

.. ipython:: python
    :okexcept:
    :okwarning:

    freq_kwargs = {**kw, **dict(gamma=ds.gamma, sigma_a=sa, sigma_b=sb, alpha=ds.alpha3)}
    method3 = construct_partition(freq_name, dir_name, freq_kwargs, dir_kwargs)
    method3 = method3.max(dim="part")


Plotting to compare

.. ipython:: python
    :okexcept:
    :okwarning:

    # Concat and plot
    dsall = xr.concat([dset.efth, method1, method2, method3], dim="fit")
    dsall["fit"] = ["Original", "Method 1", "Method 2", "Method 3"]
    dsall.spec.plot(
        figsize=(9, 9),
        col="fit",
        col_wrap=2,
        logradius=True,
        rmax=0.5,
        add_colorbar=False,
        show_theta_labels=False,
    );

    @savefig compare_stefan_methods.png
    plt.draw()