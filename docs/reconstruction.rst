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


.. _`Bunney et al. (2014)`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
