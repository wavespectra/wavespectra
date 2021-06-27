import matplotlib
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


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
    return darray


def polar_plot(
        da,
        kind="pcolormesh",
        rmin=0,
        rmax=None,
        add_labels=False,
        normalised=False,
        **kwargs
    ):
    rname = attrs.FREQNAME
    thetaname = attrs.DIRNAME
    
    # Convert directions to radians
    dtmp = _polar_dir(da)
    dtmp = dtmp.assign_coords({thetaname: np.deg2rad(dtmp[thetaname].values)})

    # Normalise energy density
    if normalised:
        dtmp /= dtmp.max()

    kwargs["x"] = thetaname
    kwargs["y"] = rname
    kwargs["add_labels"] = add_labels

    subplot_kws = {
        "projection": kwargs.pop("projection", "polar"),
        "theta_direction": kwargs.pop("theta_direction", -1),
        "theta_offset": kwargs.pop("theta_offset", np.deg2rad(90)),
    }

    pobj = getattr(dtmp.plot, kind)(subplot_kws=subplot_kws, **kwargs)

    # Set axes properties
    # pobj.colorbar.set_ticks(CONTOUR_LEVELS)
    # pobj.colorbar.set_ticklabels(CONTOUR_LEVELS)

    axes = pobj.axes
    # import ipdb; ipdb.set_trace()
    if not isinstance(pobj, xr.plot.facetgrid.FacetGrid):
        # axes.set_rscale("log")
        axes.set_rmin(rmin or dtmp.freq.min())
        axes.set_rmax(rmax or dtmp.freq.max())
        
        # axes.set_rlim((0.04, 0.4))
        import ipdb; ipdb.set_trace()
        axes.set_rscale("functionlog")
        axes.set_rticks([0.1, 0.2, 0.3, 0.4])

    return pobj


import matplotlib.pyplot as plt
from wavespectra import read_swan

dset = read_swan("/source/fork/wavespectra/tests/sample_files/swanfile.spec")
dset = dset.isel(lat=0, lon=0, drop=True)
ds = dset.isel(time=-1, drop=True)

ds = ds.where(ds > 1e-3, 1e-3)

# fig = plt.figure()
# polar_plot(ds.efth, rmax=0.2, cmap="coolwarm", add_labels=True)

fig = plt.figure()
polar_plot(
    ds.efth,
    kind="contourf",
    rmax=0.4,
    cmap="RdBu_r",
    add_labels=True,
    # normalised=True,
    levels=CONTOUR_LEVELS,
    # levels=(0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
    extend="max",
    # norm=matplotlib.colors.LogNorm(),
)

plt.show()
