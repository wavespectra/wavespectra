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


Partition and reconstruct
-------------------------

Reconstruction methods can be evaluated by partitioning existing spectra, fitting
two-dimensional shapes for each partition and recombining.

The example below uses the :meth:`~wavespectra.directional.cartwright` spreading and the
:meth:`~wavespectra.fit_jonswap` fitting with default values for :math:`\sigma_a=0.07`
and :math:`\sigma_b=0.09`, :math:`\gamma` calculated from the :meth:`~wavespectra.SpecArray.gamma`
method and :math:`\alpha` calculated from the :meth:`~wavespectra.SpecArray.alpha` method.

The spectrum is reconstructed by taking the :math:`\max{Ed}` among all partitions for each spectral bin.

.. ipython:: python
    :okexcept:
    :okwarning:

    ds = read_ww3("_static/ww3file.nc").isel(time=0, site=0, drop=True).sortby("dir")

    # Partitioning
    dspart = ds.spec.partition(ds.wspd, ds.wdir, ds.dpt).load()

    # Integrated parameters partitions
    dsparam = dspart.spec.stats(["fp", "dm", "dspr", "gamma", "alpha"])
    dsparam["dpt"] = ds.dpt.expand_dims({"part": dspart.part})

    # Construct spectra for partitions
    fit_kwargs = {
        "freq": ds.freq,
        "fp": dsparam.fp,
        "gamma": dsparam.gamma,
        "alpha": dsparam.alpha,
        "sigma_a": 0.07,
        "sigma_b": 0.09
    }
    dir_kwargs = {"dir": ds.dir, "dm": dsparam.dm, "dspr": dsparam.dspr}
    efth_part = construct_partition("fit_jonswap", "cartwright", fit_kwargs, dir_kwargs)

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

    # Reading and partitioning existing spectrum
    dset = read_ww3("_static/ww3file.nc").isel(time=0, site=-1, drop=True).sortby("dir")
    dsetp = dset.spec.partition(dset.wspd, dset.wdir, dset.dpt)

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
    fit_name = "fit_jonswap"
    kw = dict(freq=dset.freq, fp=ds.fp)


Reconstruct from method 1

.. ipython:: python
    :okexcept:
    :okwarning:

    fit_kwargs = {**kw, **dict(gamma=3.3, sigma_a=0.7, sigma_b=0.9, alpha=ds.alpha)}
    method1 = construct_partition(fit_name, dir_name, fit_kwargs, dir_kwargs)
    method1 = method1.max(dim="part")


Reconstruct from method 2

.. ipython:: python
    :okexcept:
    :okwarning:

    sa = ds.gw.where(ds.gw >= 0.04, 0.04).where(ds.gw <= 0.09, 0.09)
    sb = sa + 0.1
    fit_kwargs = {**kw, **dict(gamma=ds.gamma, sigma_a=sa, sigma_b=sb, alpha=ds.alpha)}
    method2 = construct_partition(fit_name, dir_name, fit_kwargs, dir_kwargs)
    method2 = method2.max(dim="part")


Reconstruct from method 3

.. ipython:: python
    :okexcept:
    :okwarning:

    fit_kwargs = {**kw, **dict(gamma=ds.gamma, sigma_a=sa, sigma_b=sb, alpha=ds.alpha3)}
    method3 = construct_partition(fit_name, dir_name, fit_kwargs, dir_kwargs)
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


Reconstruct CLI
---------------

Command line interface are available to reconstruct spectra.

.. code::

    $ wavespectra reconstruct --help
    Usage: wavespectra reconstruct [OPTIONS] COMMAND [ARGS]...

    Options:
    --help  Show this message and exit.

    Commands:
    spectra  Partition and reconstruct spectra from file.


Partition and reconstruct spectra from file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    $ wavespectra reconstruct spectra --help
    Usage: wavespectra reconstruct spectra [OPTIONS] INFILE OUTFILE

    Partition and reconstruct spectra from file.

    Options:
    -f, --fit_name TEXT        Fit function  [default: fit_jonswap]
    -d, --dir_name TEXT        Spread function  [default: cartwright]
    -m, --method_combine TEXT  Method to combine partitions  [default: max]
    -s, --swells INTEGER       Swell partitions to keep  [default: 6]
    -r, --reader TEXT          Spectra file reader  [default: read_ww3]
    -c, --chunks TEXT          chunks dictionary to chunk dataset  [default: {}]
    --help                     Show this message and exit.


.. admonition:: TODO
    :class: note

    * Compare Gaussian fits using the :math:`\sigma` parameter from Bunney's based :math:`Tm01` and :math:`Tm02` and the Gaussian least square parameter :math:`g_w` in WW3.
    * Review Bunney's skewed spread function.
    * Finalise the construct API.

.. _`Bunney et al. (2014)`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
