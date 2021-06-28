import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


RADII_TICKS = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4]) * 1000.0)
RADII_LABELS = [".05", "0.1", "0.2", "0.3", "0.4Hz"]
CBAR_TICKS = [1e-2, 1e-1, 1e0]
LOG_CONTOUR_LEVELS = np.logspace(np.log10(0.005), np.log10(1) , 14)
DNAME = attrs.DIRNAME
FNAME = attrs.FREQNAME
SUPPORTED_KIND = ["pcolormesh", "contourf", "contour"]


def _polar_dir(darray, kind):
    """Wrap directions and convert to radians for polar plot.

    Args:
        darray (DataArray): DataArray object to plot.
        kind (str): Plot type.

    """
    # Sort and convert to radians
    darray = darray.sortby(DNAME)
    # Close circle if not pcolor type (pcolor already closes it)
    if kind not in ["pcolor", "pcolormesh"]:
        if darray[DNAME][0] % 360 != darray[DNAME][-1] % 360:
            dd = np.diff(darray[DNAME]).mean()
            closure = darray.isel(**{DNAME: 0})
            closure = closure.assign_coords({DNAME: darray[DNAME][-1] + dd})
            darray = xr.concat((darray, closure), dim=DNAME)
    # Convert to radians
    darray = darray.assign_coords({DNAME: np.deg2rad(darray[DNAME])})
    return darray


def _set_labels(darr, normalised):
    """Redefine attributes to show in plot labels."""
    darr[FNAME].attrs["standard_name"] = "Wave frequency ($Hz$)"
    darr[DNAME].attrs["standard_name"] = "Wave direction ($degree$)"
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
        show_theta_labels=True,
        show_radii_labels=True,
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
        - show_theta_labels (bool): Show direction tick labels.
        - show_radii_labels (bool): Show radii tick labels.
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
        - If normalised==True, contourf uses a logarithmic colour scale by default.
        - Plot and axes can be redefined from the returned xarray object.
        - Xarray uses the `sharex`, `sharey` args to control which panels receive axis
          labels. In order to set labels for all panels, set these to `False`.

    """
    if kind not in SUPPORTED_KIND:
        raise NotImplementedError(
            f"Wavespectra only supports the following kinds of plots: {SUPPORTED_KIND}"
        )

    # Set kwargs for plot
    kwargs = {**kwargs, **{"x": DNAME, "y": FNAME}}
    default_subplot_kws = {
        "projection": kwargs.pop("projection", "polar"),
        "theta_direction": kwargs.pop("theta_direction", -1),
        "theta_offset": kwargs.pop("theta_offset", np.deg2rad(90)),
        "rmin": rmin,
        "rmax": rmax,
    }
    subplot_kws = {**default_subplot_kws, **kwargs.get("subplot_kws", {})}

    # Adjust directions for polar axis
    dtmp = _polar_dir(da, kind)

    # Redefine labels
    dtmp = _set_labels(dtmp, normalised)

    # Normalise energy density
    if normalised:
        dtmp /= dtmp.max()
        dtmp = dtmp.where(dtmp >= efth_min, efth_min)

    # Convert to log
    if as_log10:
        dtmp = dtmp.assign_coords({FNAME: np.log10(dtmp[FNAME] * 1000)})
        rmin = np.log10(subplot_kws["rmin"] * 1000)
        rmax = np.log10(subplot_kws["rmax"] * 1000)
        # Use by default log colours in contourf if normalised
        if normalised and kind == "contourf" and "levels" not in kwargs:
            kwargs.update({"levels": LOG_CONTOUR_LEVELS})

    # Call plotting function
    pobj = getattr(dtmp.plot, kind)(subplot_kws=subplot_kws, **kwargs)

    if isinstance(pobj, xr.plot.facetgrid.FacetGrid):
        axes = list(pobj.axes.ravel())
        cbar = pobj.cbar
    else:
        axes = [pobj.axes]
        cbar = pobj.colorbar

    # Adjusting axes
    for ax in axes:
        ax.set_rgrids(
            radii_ticks, radii_labels, radii_labels_angle, size=radii_label_size,
        )
        ax.set_rmax(rmax)
        ax.set_rmin(rmin)

        # Disable labels as they are drawn on top of ticks
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Disable or not tick labels
        if show_theta_labels is False:
            ax.set_xticklabels("")
        if show_radii_labels is False:
            ax.set_yticklabels("")

    # Adjusting colorbar
    if cbar is not None:
        cbar.set_ticks(cbar_ticks)

    return pobj
