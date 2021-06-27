import matplotlib
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


RADII_TICKS = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4]) * 1000.0)
RADII_LABELS = [".05", "0.1", "0.2", "0.3", "0.4Hz"]
CBAR_TICKS = [1e-2, 1e-1, 1e0]
LOG_LEVELS = np.array(
    [
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.06,
        0.08,
        0.13,
        0.19,
        0.28,
        0.43,
        0.64,
        0.96,
        1.0,
    ]
)


def _polar_dir(darray):
    """Wrap directions and convert to radians for polar plot."""
    dname = attrs.DIRNAME
    # Sort and convert to radians
    darray = darray.sortby(dname)
    # Close circle
    if darray[dname][0] % 360 != darray[dname][-1] % 360:
        dd = np.diff(darray[dname]).mean()
        closure = darray.isel(**{dname: 0})
        closure = closure.assign_coords({dname: darray[dname][-1] + dd})
        darray = xr.concat((darray, closure), dim=dname)
    # Convert to radians
    darray = darray.assign_coords({dname: np.deg2rad(darray[dname])})
    return darray


def _set_labels(darr, normalised):
    """Redefine attributes to show in plot labels."""
    fname = attrs.FREQNAME
    dname = attrs.DIRNAME
    darr[fname].attrs["standard_name"] = "Wave frequency ($Hz$)"
    darr[dname].attrs["standard_name"] = "Wave direction ($degree$)"
    if normalised:
        darr.attrs["standard_name"] = "Normalised Energy Density"
        darr.attrs["units"] = ""
    else:
        darr.attrs["standard_name"] = "Energy Density"
        darr.attrs["units"] = "$m^{2}s/deg$"
    return darr


def polar_plot(
        da,
        kind="pcolormesh",
        rmin=0.03,
        rmax=0.5,
        normalised=False,
        as_log10=True,
        as_period=False,
        radii_ticks=RADII_TICKS,
        radii_labels=RADII_LABELS,
        radii_labels_angle=22.5,
        radii_label_size=8,
        cbar_ticks=CBAR_TICKS,
        efth_min=1e-3,
        **kwargs
    ):
    """Plot spectra in polar axis.

    Args:
        - da (DataArray): Wavespectra DataArray.
        - kind (str): The kind of plot to produce, e.g. `contourf`, `pcolormesh`.
        - rmin (float): Minimum value to clip the radius axis.
        - rmax (float): Maximum value to clip the radius axis.
        - normalised (bool): Plot normalised efth between 0 and 1.
        - as_log10 (bool): Plot efth on a log radius.
        - as_period (bool): Set radii as wave period instead of frequency.
        - radii_ticks (array): Tick values for radii.
        - radii_labels (array): Ticklabel values for radii.
        - radii_labels_angle (float): Polar angle at which radii labels are positioned.
        - radii_label_size (float): Fontsize for radii labels.
        - cbar_ticks (array): Tick values for colorbar.
        - efth_min (float): Mask energy density below this value.
        - kwargs: All extra kwargs are passed to the plotting method defined by `kind`.

    Returns:
        - pobj: The xarray object returned by calling `da.plot.{kind}(**kwargs)`.

    Note:
        Plot and axes can be redefined from the xarray object returned in this function.

    """
    fname = attrs.FREQNAME
    dname = attrs.DIRNAME

    # Set kwargs for plot
    kwargs = {**kwargs, **{"x": dname, "y": fname}}
    default_subplot_kws = {
        "projection": kwargs.pop("projection", "polar"),
        "theta_direction": kwargs.pop("theta_direction", -1),
        "theta_offset": kwargs.pop("theta_offset", np.deg2rad(90)),
        "rmin": rmin,
        "rmax": rmax,
    }
    subplot_kws = {**default_subplot_kws, **kwargs.get("subplot_kws", {})}

    # Adjust directions for polar axis
    dtmp = _polar_dir(da)

    # Redefine labels
    dtmp = _set_labels(dtmp, normalised)

    # Normalise energy density
    if normalised:
        dtmp /= dtmp.max()
        dtmp = dtmp.where(dtmp >= efth_min, efth_min)

    # Convert to log
    if as_log10:
        dtmp = dtmp.assign_coords({fname: np.log10(dtmp[fname] * 1000)})
        rmin = np.log10(subplot_kws["rmin"] * 1000)
        rmax = np.log10(subplot_kws["rmax"] * 1000)

    # import ipdb; ipdb.set_trace()
    # Call plotting function
    pobj = getattr(dtmp.plot, kind)(subplot_kws=subplot_kws, **kwargs)

    # Adjusting axis
    ax = pobj.axes
    ax.set_rgrids(radii=radii_ticks, labels=radii_labels, angle=radii_labels_angle, size=radii_label_size)
    ax.set_rmax(rmax)
    ax.set_rmin(rmin)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Adjusting colorbar
    pobj.colorbar.set_ticks(cbar_ticks)

    return pobj


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from wavespectra import read_swan

    dset = read_swan("/source/fork/wavespectra/tests/sample_files/swanfile.spec")
    dset = dset.isel(lat=0, lon=0, drop=True)
    ds = dset.isel(time=-1, drop=True)


    fig = plt.figure()
    pobj = polar_plot(
        da=ds.efth,
        kind="contourf",
        rmin=0.03,
        rmax=0.5,
        normalised=True,
        as_log10=True,
        as_period=False,
        radii_ticks=RADII_TICKS,
        radii_labels=RADII_LABELS,
        radii_labels_angle=22.5,
        radii_label_size=8,
        cbar_ticks=CBAR_TICKS,
        efth_min=1e-3,

        cmap="RdBu_r",
        extend="neither",
        levels=LOG_LEVELS,
    )

    plt.show()
