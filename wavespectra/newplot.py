import matplotlib
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


RADII_TICKS = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4]) * 1000.0)
RADII_LABELS = [".05", "0.1", "0.2", "0.3", "0.4Hz"]
CONTOUR_LEVELS = np.array(
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


def polar_plot(
        da,
        kind="pcolormesh",
        rmin=0.03,
        rmax=0.5,
        normalised=False,
        as_log10=True,
        radii_labels_angle=22.5,
        radii_ticks=RADII_TICKS,
        radii_labels=RADII_LABELS,
        radii_label_size=8,
        **kwargs
    ):
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

    # Normalise energy density
    if normalised:
        dtmp /= dtmp.max()
        # dtmp = dtmp.where(dtmp >= 1e-3, 1e-3)

    # Convert to log
    if as_log10:
        dtmp = dtmp.assign_coords({fname: np.log10(dtmp[fname] * 1000)})
        rmin = np.log10(subplot_kws["rmin"] * 1000)
        rmax = np.log10(subplot_kws["rmax"] * 1000)

    # Call plotting function
    pobj = getattr(dtmp.plot, kind)(subplot_kws=subplot_kws, **kwargs)

    # Adjusting axis
    ax = pobj.axes
    ax.set_rgrids(radii=radii_ticks, labels=radii_labels, angle=radii_labels_angle, size=radii_label_size)
    ax.set_rmax(rmax)
    ax.set_rmin(rmin)
    # ax.tick_params(labelsize=6)

    # # Set axes properties
    # # pobj.colorbar.set_ticks(CONTOUR_LEVELS)
    # # pobj.colorbar.set_ticklabels(CONTOUR_LEVELS)

    # axes = pobj.axes
    # # import ipdb; ipdb.set_trace()
    # if not isinstance(pobj, xr.plot.facetgrid.FacetGrid):
    #     # axes.set_rscale("log")
    #     # axes.set_rmin(rmin or dtmp.freq.min())
    #     # axes.set_rmax(rmax or dtmp.freq.max())
        
    #     axes.set_rmin(rmin)
    #     axes.set_rmax(rmax)
    #     # import ipdb; ipdb.set_trace()
    #     if as_log10:
    #         rT = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4]) * 1000.0)
    #         axes.set_rgrids(rT, [str(v) for v in rT], size=7, horizontalalignment="right", angle=155)
    #         axes.set_rscale("log")
    #     # axes.set_rticks([0.1, 0.2, 0.3, 0.4])

    #     # # Format rticks
    #     # axes.set_rgrids(
    #     #     rT, labels=rTL, size=7, horizontalalignment="right", angle=155
    #     # )

    return pobj


import matplotlib.pyplot as plt
from wavespectra import read_swan

dset = read_swan("/source/fork/wavespectra/tests/sample_files/swanfile.spec")
dset = dset.isel(lat=0, lon=0, drop=True)
ds = dset.isel(time=-1, drop=True)

ds = ds.where(ds > 1e-3, 1e-3)


fig = plt.figure()
pobj = polar_plot(
    ds.efth,
    kind="contourf",
    # rmax=0.4,
    cmap="RdBu_r",
    # cmap="pink",
    # normalised=True,
    levels=CONTOUR_LEVELS,
    # levels=(0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
    extend="neither",
    # norm=matplotlib.colors.LogNorm(),
)

plt.show()
