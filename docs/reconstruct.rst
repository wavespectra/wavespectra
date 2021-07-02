Spectra reconstruction
______________________


.. ipython:: python
    :okexcept:
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from wavespectra import read_ww3
    from wavespectra import fit_pierson_moskowitz, fit_jonswap, fit_tma, fit_gaussian
    from wavespectra.directional_distribution import cartwright, bunney
    from wavespectra.construct import construct_partition


Partition spectrum and reconstruct from Jonswap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. admonition:: Quick reconstruction example from existing WW3 spectrum
    :class: note

    * Use Jonswap only (no Gaussian fit).
    * No calculation for alpha (using default of 0.0081).
    * Use default :math:`\sigma_a=0.07`, :math:`\sigma_b=0.09`. 
    * Use :math:`\gamma` calculated from wavespectra.
    * Use Cartwright cosine-squared distribution (no skewed wind sea spreading).


.. ipython:: python
    :okexcept:
    :okwarning:

    ds = read_ww3("_static/ww3file.nc").isel(time=0, site=0, drop=True).sortby("dir")

    # Partition
    dspart = ds.spec.partition(ds.wspd, ds.wdir, ds.dpt).load()

    # Parameters
    dsparam = dspart.spec.stats(["hs", "tp", "dm", "tm01", "tm02", "dspr", "gamma"])
    dsparam["dpt"] = ds.dpt.expand_dims({"part": dspart.part})

    # Construct Jonswap for all partitions from parameters
    fit_name = "fit_jonswap"
    fit_kwargs = {
        "freq": ds.freq,
        "hs": dsparam.hs,
        "tp": dsparam.tp,
        "gamma": dsparam.gamma,
        "alpha": 0.0081,
        "sigma_a": 0.07,
        "sigma_b": 0.09
    }
    dir_name = "cartwright"
    dir_kwargs = {"dir": ds.dir, "dm": dsparam.dm, "dspr": dsparam.dspr}
    efth_part = construct_partition(fit_name, fit_kwargs, dir_kwargs, dir_name)

    # Reconstruct from the max along the part dim
    efth_max = efth_part.max(dim="part")
    ds_comp = xr.concat([ds.efth, efth_max], dim="spectype")
    ds_comp["spectype"] = ["Original", "Reconstructed"]

    ds_comp.spec.plot(
        normalised=True,
        as_period=False,
        logradius=True,
        figsize=(8,6),
        show_theta_labels=False,
        add_colorbar=False,
        col="spectype",
    );

    @savefig original_vs_reconstructed.png
    plt.draw()


.. admonition:: TODO
    :class: note

    Swell gaussian fit
        * Stefan defines :math:`\sigma_a` (the left width of the peak enhancement factor :math:`\gamma`) from the Gaussian spread parameter
          with :math:`\sigma_b=\sigma_{a}+0.1`, and uses them in the Jonswap formula.
        * I have currently implemented the formula as given in Bunney's, not sure if they will be equivalent.
        * Check these, compare, decide which one to use. The gaussian parameter is not implemented yet in wavespectra (but is in WW3 output files).
    Skewed sea wind directional distribution
        * This is not implemented from Stefan's code as far as I can tell.
        * Some parameters as defined in the paper are not too clear to me so need to find those out.
    Finalise the construct API
        * Define function to choose the fittings based on criteria.
        * Define functions to construct from WW3 and SWAN parameters files.

.. _`Bunney et al. (2014)`: https://www.icevirtuallibrary.com/doi/abs/10.1680/fsts.59757.114
