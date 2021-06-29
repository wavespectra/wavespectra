import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


LOG_FACTOR = 1e3
RADII_FREQ_TICKS_LOG = np.log10(np.array([0.05, 0.1, 0.2, 0.3, 0.4]) * LOG_FACTOR)
RADII_FREQ_TICKS_LIN = np.arange(0.1, 1.1, 0.1)
RADII_PER_TICKS_LOG = np.log10((np.array([20, 10, 5, 3, 2])) * LOG_FACTOR)
RADII_PER_TICKS_LIN = np.arange(5, 30, 5)
CBAR_TICKS = [1e-2, 1e-1, 1e0]
LOG_CONTOUR_LEVELS = np.logspace(np.log10(0.005), np.log10(1) , 14)
SUPPORTED_KIND = ["pcolormesh", "contourf", "contour"]


class WavePlot:
    """Plot spectra in polar axis.

    Args:
        - darr (DataArray): Wavespectra DataArray.
        - kind (str): Plot kind, one of (`contourf`, `contour`, `pcolormesh`).
        - normalised (bool): Show efth normalised between 0 and 1.
        - as_log10 (bool): Show efth on a log radius.
        - as_period (bool): Set radii as wave period instead of frequency.
        - rmin (float): Minimum value to clip the radius axis.
        - rmax (float): Maximum value to clip the radius axis.
        - show_theta_labels (bool): Show direction tick labels.
        - show_radii_labels (bool): Show radii tick labels.
        - radii_ticks (array): Tick values for radii.
        - radii_labels_angle (float): Polar angle at which radii labels are positioned.
        - radii_label_size (float): Fontsize for radii labels.
        - cbar_ticks (array): Tick values for colorbar.
        - cmap (str, obj): Colormap to use.
        - efth_min (float): Clip energy density below this value.
        - kwargs: All extra kwargs are passed to the plotting method defined by `kind`.

    Returns:
        - pobj: The xarray object returned by calling `da.plot.{kind}(**kwargs)`.

    """
    def __init__(
        self,
        darr,
        kind="contourf",
        normalised=False,
        as_log10=True,
        as_period=False,
        rmin=None,
        rmax=None,
        show_theta_labels=True,
        show_radii_labels=True,
        radii_ticks=None,
        radii_labels_angle=22.5,
        radii_label_size=8,
        cbar_ticks=CBAR_TICKS,
        cmap="RdBu_r",
        extend="neither",
        efth_min=1e-3,
        **kwargs
    ):
        self.kind = kind
        self.normalised = normalised
        self.as_log10 = as_log10
        self.as_period = as_period
        self.show_theta_labels = show_theta_labels
        self.show_radii_labels = show_radii_labels
        self.radii_labels_angle = radii_labels_angle
        self.radii_label_size = radii_label_size
        self.cbar_ticks = cbar_ticks
        self.cmap = cmap
        self.extend = extend
        self.efth_min = efth_min

        # Attributes set based on other attributes
        self._darr = darr
        self._rmin = rmin
        self._rmax = rmax
        self._radii_ticks = radii_ticks
        self._kwargs = kwargs

        self._validate()

    def __repr__(self):
        s = f"<Waveplot {self.kind}>"
        for attr in ["normalised", "as_log10", "as_period"]:
            if getattr(self, attr):
                s = s.replace(">", f" {attr.split('_')[-1]}>")
        return s

    @property
    def dname(self):
        return attrs.DIRNAME

    @property
    def fname(self):
        return attrs.FREQNAME

    @property
    def darr(self):
        _darr = self._polar_dir(self._darr)
        _darr = self._set_labels(_darr)
        if self.as_period:
            _darr = self._to_period(_darr)
        if self.as_log10:
            _darr = self._to_log10(_darr)
        return _darr

    @property
    def rmin(self):
        """The radius centre."""
        if self._rmin is None:
            return float(self.darr[self.fname].min())
        if self.as_log10 and self._rmin > 0:
            return np.log10(self._rmin * LOG_FACTOR)
        else:
            return self._rmin

    @property
    def rmax(self):
        """The radius edge."""
        if self._rmax is None:
            return float(self.darr[self.fname].max())
        if self.as_log10 and self._rmax > 0:
            return np.log10(self._rmax * LOG_FACTOR)
        else:
            return self._rmax

    @property
    def radii_ticks(self):
        """Tick locations for the radii axis."""
        if self._radii_ticks is not None:
            return self._radii_ticks
        if self.as_log10:
            if self.as_period:
                self._radii_ticks = RADII_PER_TICKS_LOG
            else:
                self._radii_ticks = RADII_FREQ_TICKS_LOG
        else:
            if self.as_period:
                self._radii_ticks = RADII_PER_TICKS_LIN
            else:
                self._radii_ticks = RADII_FREQ_TICKS_LIN
        # Clipping to radius limits
        self._radii_ticks = np.unique(self._radii_ticks.clip(self.rmin, self.rmax))
        return self._radii_ticks

    @property
    def radii_ticklabels(self):
        """Tick labels for the radii axis."""
        units = self.darr[self.fname].attrs.get("units", "Hz")
        if self.as_log10:
            ticks = 10 ** self.radii_ticks / 1000
        else:
            ticks = self.radii_ticks
        ticklabels = [f"{v:g}" for v in ticks]
        ticklabels[-1] = ticklabels[-1] + units
        return ticklabels

    @property
    def kwargs(self):
        _kwargs = {
            **self._kwargs, **{"x": self.dname, "y": self.fname, "cmap": self.cmap}
        }
        if self.kind == "contourf" and "extend" not in _kwargs:
            _kwargs.update({"extend": "neither"})
        if self.normalised and "contour" in self.kind and "level" not in _kwargs:
            _kwargs.update({"levels": LOG_CONTOUR_LEVELS})
        return _kwargs

    def _validate(self):
        """Validate input arguments."""
        if self.kind not in SUPPORTED_KIND:
            raise NotImplementedError(
                f"Wavespectra only supports the following kinds: {SUPPORTED_KIND}"
            )

    def _polar_dir(self, darray):
        """Sort, wrap and convert directions to radians for polar plot."""
        # Sort directions
        darray = darray.sortby(self.dname)
        if self.kind not in ["pcolor", "pcolormesh"]:
            # Close circle if not pcolor type (pcolor already closes it)
            if darray[self.dname][0] % 360 != darray[self.dname][-1] % 360:
                dd = np.diff(darray[self.dname]).mean()
                closure = darray.isel(**{self.dname: 0})
                closure = closure.assign_coords({self.dname: darray[self.dname][-1] + dd})
                darray = xr.concat((darray, closure), dim=self.dname)
        # Convert to radians
        darray = darray.assign_coords({self.dname: np.deg2rad(darray[self.dname])})
        return darray

    def _set_labels(self, darr):
        """Redefine attributes to show in plot labels."""
        darr[self.fname].attrs["standard_name"] = "Wave frequency ($Hz$)"
        darr[self.dname].attrs["standard_name"] = "Wave direction ($degree$)"
        if self.normalised:
            darr.attrs["standard_name"] = "Normalised Energy Density"
            darr.attrs["units"] = ""
        else:
            darr.attrs["standard_name"] = "Energy Density"
            darr.attrs["units"] = "$m^{2}s/deg$"
        return darr

    def _to_period(self, darr):
        darr = darr.assign_coords({self.fname: 1 / darr[self.fname]})
        darr[self.fname].attrs.update(
            {"standard_name": "sea_surface_wave_period", "units": "s"}
        )
        return darr

    def _to_log10(self, darr):
        fattrs = darr[self.fname].attrs
        dattrs = darr[self.dname].attrs
        sattrs = darr.attrs
        darr = darr.assign_coords(
            {self.fname: np.log10(darr[self.fname] * LOG_FACTOR)}
        )
        darr.attrs = sattrs
        darr[self.fname].attrs = fattrs
        darr[self.dname].attrs = dattrs
        return darr

    def __call__(self):
        """Execute plotting method."""

        # kwargs for polar axis
        default_subplot_kws = {
            "projection": self._kwargs.pop("projection", "polar"),
            "theta_direction": self._kwargs.pop("theta_direction", -1),
            "theta_offset": self._kwargs.pop("theta_offset", np.deg2rad(90)),
        }
        subplot_kws = {**default_subplot_kws, **self._kwargs.get("subplot_kws", {})}

        # Call plotting function
        pobj = getattr(self.darr.plot, self.kind)(subplot_kws=subplot_kws, **self.kwargs)

        # Adjusting axes
        if isinstance(pobj, xr.plot.facetgrid.FacetGrid):
            axes = list(pobj.axes.ravel())
            cbar = pobj.cbar
        else:
            axes = [pobj.axes]
            cbar = pobj.colorbar

        for ax in axes:
            # import ipdb; ipdb.set_trace()
            ax.set_rgrids(
                radii=self.radii_ticks,
                labels=self.radii_ticklabels,
                angle=self.radii_labels_angle,
                size=self.radii_label_size,
            )
            ax.set_rmax(self.rmax)
            ax.set_rmin(self.rmin)

            # Disable labels as they are drawn on top of ticks
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Disable or not tick labels
            if self.show_theta_labels is False:
                ax.set_xticklabels("")
            if self.show_radii_labels is False:
                ax.set_yticklabels("")

        # Adjusting colorbar
        if cbar is not None:
            cbar.set_ticks(self.cbar_ticks)

        return pobj


def polar_plot(*args, **kargs):
    """Plot spectra in polar axis.

    Args:
        - darr (DataArray): Wavespectra DataArray.
        - kind (str): Plot kind, one of (`contourf`, `contour`, `pcolormesh`).
        - normalised (bool): Show efth normalised between 0 and 1.
        - as_log10 (bool): Show efth on a log radius.
        - as_period (bool): Set radii as wave period instead of frequency.
        - rmin (float): Minimum value to clip the radius axis.
        - rmax (float): Maximum value to clip the radius axis.
        - show_theta_labels (bool): Show direction tick labels.
        - show_radii_labels (bool): Show radii tick labels.
        - radii_ticks (array): Tick values for radii.
        - radii_labels_angle (float): Polar angle at which radii labels are positioned.
        - radii_label_size (float): Fontsize for radii labels.
        - cbar_ticks (array): Tick values for colorbar.
        - cmap (str, obj): Colormap to use.
        - efth_min (float): Clip energy density below this value.
        - kwargs: All extra kwargs are passed to the plotting method defined by `kind`.

    Returns:
        - pobj: The xarray object returned by calling `da.plot.{kind}(**kwargs)`.

    """
    wp = WavePlot(*args, **kargs)
    return wp()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from wavespectra import read_swan
    dset = read_swan("/source/fork/wavespectra/tests/sample_files/swanfile.spec").isel(lon=0, lat=0, drop=True)
    ds = dset.isel(time=0, drop=True)

    # self = WavePlot(
    #     ds.efth.spec._obj,
    #     normalised=False,
    #     cmap="pink_r",
    #     levels=15,
    #     rmin=None,
    #     rmax=0.3,
    #     as_log10=False,
    #     as_period=False,
    #     radii_ticks=[0.05, 0.1, 0.15, 0.2, 0.25],
    #     )

    # fig = plt.figure()
    # self()
    # plt.show()


    pobj = polar_plot(
        ds.efth.spec._obj,
        normalised=True,
        # cmap="pink_r",
        # levels=15,
        rmin=None,
        rmax=None,
        as_log10=True,
        as_period=True,
        # radii_ticks=[0.05, 0.1, 0.15, 0.2, 0.25],
    )

    # fig = plt.figure()
    # self()
    plt.show()