"""Spectra plotting."""
import functools
import numpy as np
import matplotlib.pyplot as plt
from xarray.plot.plot import _plot2d as _xarray_plot2d
from xarray.plot.plot import _PlotMethods as _XarrayPlotMethods

from wavespectra.core.attributes import attrs


def _plot2d(plotfunc, **kwargs):
    print("AAA")
    primitive = _xarray_plot2d(plotfunc, **kwargs)

    # For use as DataArray.plot.plotmethod
    @functools.wraps(newplotfunc)
    def plotmethod(
        _PlotMethods_obj,
        x=None,
        y=None,
        figsize=None,
        size=None,
        aspect=None,
        ax=None,
        row=None,
        col=None,
        col_wrap=None,
        xincrease=True,
        yincrease=True,
        add_colorbar=None,
        add_labels=True,
        vmin=None,
        vmax=None,
        cmap=None,
        colors=None,
        center=None,
        robust=False,
        extend=None,
        levels=None,
        infer_intervals=None,
        subplot_kws=None,
        cbar_ax=None,
        cbar_kwargs=None,
        xscale=None,
        yscale=None,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        norm=None,
        **kwargs
    ):
        """
        The method should have the same signature as the function.

        This just makes the method work on Plotmethods objects,
        and passes all the other arguments straight through.
        """
        allargs = locals()
        allargs["darray"] = _PlotMethods_obj._da
        allargs.update(kwargs)
        for arg in ["_PlotMethods_obj", "newplotfunc", "kwargs"]:
            del allargs[arg]
        return newplotfunc(**allargs)

    # Add to class _PlotMethods
    setattr(_PlotMethods, plotmethod.__name__, plotmethod)

    return newplotfunc

@_plot2d
def contourf(x, y, z, ax, **kwargs):
    """Filled contour polar plot of 2d SpecArray.

    Wraps :func:`matplotlib:matplotlib.pyplot.contourf`.

    """
    dgrid, rgrid = np.meshgrid(np.deg2rad(x), 1./y)
    zpos = np.log10(np.ma.masked_where(z==0, z).filled(1e-5))

    kwargs.update({
        "vmin": zpos.min(),
        "vmax": zpos.max(),
        "levels": np.linspace(float(zpos.min()), float(zpos.max()), 15),
        "norm": None,
    })

    primitive = ax.contourf(x, y, zpos, **kwargs)
    return primitive


class _PlotMethods(_XarrayPlotMethods):    
    def __init__(self, darray, x=None, y=None, **kwargs):
        """Modified xarray plotting object.
        
        Args:
            x (list): Array with x-coordinates.
            y (list): Array with y-coordinates.

        """
        self._x = x
        self._y = y
        import ipdb; ipdb.set_trace()
        super(_PlotMethods, self).__init__(darray, **kwargs)

#     def __call__(self, **kwargs):
#         """Define polar axis if spectra is directional."""
#         if attrs["FREQNAME"] in self._da.dims and self._da[attrs["FREQNAME"]].size > 1:
#             if attrs["DIRNAME"] in self._da.dims and self._da[attrs["DIRNAME"]].size > 1:
#                 if 'col' in kwargs.keys() or 'row' in kwargs.keys():
#                     if not 'subplot_kws' in kwargs.keys():
#                         kwargs.update(dict(
#                             subplot_kws=dict(projection = 'polar'),
#                             sharex = False,
#                             sharey = False,
#                         ))
#         r = super(_PlotMethods, self).__call__(**kwargs)
#         self.find_axes(r)
#         self.rotate_axes()
#         return r

#     def find_axes(self, r):
#         """Find axes."""
#         try:
#             self._axes = r.axes.flatten()
#         except:
#             try:
#                 self._axes = r.axes
#             except:
#                 self._axes = r
#         try:
#             iter(self._axes)
#         except:
#             self._axes = [self._axes]

#     def rotate_axes(self):
#         """Rotate polars."""
#         try:
#             for ax in self._axes:
#                 if isinstance(ax, PolarAxes):
#                     ax.set_theta_zero_location('N')
#                     ax.set_theta_direction(-1)
#         except:
#             pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from wavespectra import read_swan

    dset = read_swan("../tests/sample_files/swanfile.spec", as_site=True)

    darr = dset.isel(site=0, time=0).efth.sortby("dir")
    # darr.plot()
    contourf(darr, x="dir", y="freq")

    plt.show()