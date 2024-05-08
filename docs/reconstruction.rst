Spectra reconstruction
______________________


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
    from wavespectra import read_ww3
    @suppress
    from wavespectra import fit_pierson_moskowitz, fit_jonswap, fit_tma, fit_gaussian
    @suppress
    from wavespectra.directional import cartwright, bunney
    @suppress
    from wavespectra.construct import construct_partition


Spectra with multiple wave systems can be reconstructed by fitting spectral shapes
and directional distributions to individual wave partitions and combining them together.


The example below uses the :meth:`~wavespectra.construct.direction.cartwright`
spreading and the :meth:`~wavespectra.construct.frequency.jonswap` shape with default
values for :math:`\sigma_a=0.07` and :math:`\sigma_b=0.09`, :math:`\gamma` calculated
from the :meth:`~wavespectra.SpecArray.gamma` method and :math:`\alpha` calculated from
the :meth:`~wavespectra.SpecArray.alpha` method.

The spectrum is reconstructed by taking the :math:`\max{Ed}` among all partitions for each spectral bin.

.. ipython:: python
    :okexcept:
    :okwarning:

    ds = read_ww3("_static/ww3file.nc").isel(time=0, site=0, drop=True).sortby("dir")

    # Partitioning
    dspart = ds.spec.partition.ptm1(ds.wspd, ds.wdir, ds.dpt).load()

    # Integrated parameters partitions
    dsparam = dspart.spec.stats(["hs", "fp", "dm", "dspr", "gamma"])
    dsparam["dpt"] = ds.dpt.expand_dims({"part": dspart.part})

    # Construct spectra for partitions
    freq_kwargs = {
        "freq": ds.freq,
        "hs": dsparam.hs,
        "fp": dsparam.fp,
        "gamma": dsparam.gamma,
        "sigma_a": 0.07,
        "sigma_b": 0.09
    }
    dir_kwargs = {"dir": ds.dir, "dm": dsparam.dm, "dspr": dsparam.dspr}
    efth_part = construct_partition("jonswap", "cartwright", freq_kwargs, dir_kwargs)

    # Combine partitions from the max along the `part` dim
    efth_max = efth_part.max(dim="part")

    # Plot original and reconstructed spectra
    ds_comp = xr.concat([ds.efth, efth_max], dim="spectype")
    ds_comp["spectype"] = ["Original", "Reconstructed"]

    ds_comp.spec.plot(
        normalised=True,
        as_period=False,
        logradius=True,
        figsize=(8,4),
        show_theta_labels=False,
        add_colorbar=False,
        col="spectype",
    );

    @savefig original_vs_reconstructed.png
    plt.draw()


Partition and reconstruct
-------------------------


The :func:`~wavespectra.construct.partition_and_reconstruct` function allows
partitioning and reconstructing existing spectra in a convenient way:

.. ipython:: python
    :okexcept:
    :okwarning:

    ds = read_ww3("_static/ww3file.nc").isel(time=0, site=0, drop=True).sortby("dir")

    # Use Cartwright and Jonswap
    dsr1 = partition_and_reconstruct(
        ds,
        parts=4,
        freq_name="jonswap",
        dir_name="cartwright",
        partition_method="ptm1",
        method_combine="max",
    )

    # Asymmetric for wind sea and Cartwright for swells, Jonswap for all partitions
    dsr2 = partition_and_reconstruct(
        ds,
        parts=4,
        freq_name="jonswap",
        dir_name=["asymmetric", "cartwright", "cartwright", "cartwright",],
        partition_method="ptm1",
        method_combine="max",
    )

    # Plotting
    dsall = xr.concat([ds.efth, dsr1.efth, dsr2.efth], dim="directype")
    dsall["directype"] = ["Original", "Cartwright", "Asymmetric+Cartwright"]

    dsall.spec.plot(
        figsize=(8,4),
        show_theta_labels=False,
        add_colorbar=False,
        col="directype",
    );

    @suppress
    plt.tight_layout()

    @savefig original_vs_cartwright_vs_bunney.png
    plt.draw()


Zieger approach
----------------

.. attention::

    **Note when reviewing**

    I can't remember exactly where these Zieger methods came from, perhaps from Ron?
    Should we keep this in the docs?

    The alpha method in wavespectra is different, it is based on the slope of the high
    frequency tail, but it has some issues. Perhaps this alpha implementation should be
    used instead?


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


.. _`Bunney et al. (2014)`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
