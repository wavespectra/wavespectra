import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


LOG_FACTOR = 1e3
RADII_FREQ_TICKS_LOG = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
RADII_FREQ_TICKS_LIN = np.arange(0.1, 1.1, 0.1)
RADII_PER_TICKS_LOG = (np.array([20, 10, 5, 3, 2]))
RADII_PER_TICKS_LIN = np.arange(5, 30, 5)
CBAR_TICKS = [1e-2, 1e-1, 1e0]
LOG_CONTOUR_LEVELS = np.logspace(np.log10(0.005), np.log10(1), 14)
SUPPORTED_KIND = ["pcolormesh", "contourf", "contour"]


class WavePlot:
    """Plot spectra in polar axis.

    Args:
        - darr (DataArray): Wavespectra DataArray.
        - kind (str): Plot kind, one of (`contourf`, `contour`, `pcolormesh`).
        - normalised (bool): Show efth normalised between 0 and 1.
        - logradius (bool): Set log radii.
        - as_period (bool): Set radii as wave period instead of frequency.
        - rmin (float): Minimum value to clip the radius axis.
        - rmax (float): Maximum value to clip the radius axis.
        - show_theta_labels (bool): Show direction tick labels.
        - show_radii_labels (bool): Show radii tick labels.
        - radii_ticks (array): Tick values for radii.
        - radii_labels_angle (float): Polar angle at which radii labels are positioned.
        - radii_labels_size (float): Fontsize for radii labels.
        - cbar_ticks (array): Tick values for colorbar.
        - cmap (str, obj): Colormap to use.
        - efth_min (float): Clip energy density below this value.
        - clean_axis (bool): Remove radii and theta axis lines for clean view.
        - kwargs: All extra kwargs are passed to the plotting method defined by `kind`.

    Returns:
        - pobj: The xarray object returned by calling `da.plot.{kind}(**kwargs)`.

    """

    def __init__(
        self,
        darr,
        kind="contourf",
        normalised=True,
        logradius=True,
        as_period=False,
        rmin=None,
        rmax=None,
        show_theta_labels=True,
        show_radii_labels=True,
        radii_ticks=None,
        radii_labels_angle=22.5,
        radii_labels_size=8,
        cbar_ticks=None,
        cmap="RdBu_r",
        extend="neither",
        efth_min=1e-3,
        clean_axis=False,
        **kwargs,
    ):
        self.kind = kind
        self.normalised = normalised
        self.logradius = logradius
        self.as_period = as_period
        self.show_theta_labels = show_theta_labels
        self.show_radii_labels = show_radii_labels
        self.radii_labels_angle = radii_labels_angle
        self.radii_labels_size = radii_labels_size
        self.cmap = cmap
        self.extend = extend
        self.efth_min = efth_min
        self.clean_axis = clean_axis

        # Attributes set based on other attributes
        self._darr = darr
        self._rmin = rmin
        self._rmax = rmax
        self._radii_ticks = radii_ticks
        self._cbar_ticks = cbar_ticks
        self._kwargs = kwargs

        self._validate()

    def __repr__(self):
        s = f"<Waveplot {self.kind}>"
        for attr in ["normalised", "logradius", "as_period"]:
            if getattr(self, attr):
                s = s.replace(">", f" {attr.split('_')[-1]}>")
        return s

    def __call__(self):
        """Execute plotting method."""

        # kwargs for polar axis
        default_subplot_kws = {
            "projection": self._kwargs.pop("projection", "polar"),
            "theta_direction": self._kwargs.pop("theta_direction", -1),
            "theta_offset": self._kwargs.pop("theta_offset", np.deg2rad(90)),
        }
        subplot_kws = {**default_subplot_kws, **self._kwargs.pop("subplot_kws", {})}

        # Call plotting function
        pobj = getattr(self.darr.plot, self.kind)(
            subplot_kws=subplot_kws, **self.kwargs
        )

        # Adjusting axes
        if isinstance(pobj, xr.plot.facetgrid.FacetGrid):
            axes = list(pobj.axes.ravel())
            cbar = pobj.cbar
        else:
            axes = [pobj.axes]
            cbar = pobj.colorbar

        for ax in axes:
            ax.set_rgrids(
                radii=self.radii_ticks,
                labels=self.radii_ticklabels,
                angle=self.radii_labels_angle,
                size=self.radii_labels_size,
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

            # Clean axis
            if self.clean_axis:
                ax.set_rticks([])
                ax.set_xticks([])

        # Adjusting colorbar
        if cbar is not None:
            cbar.set_ticks(self.cbar_ticks)

        return pobj

    @property
    def dname(self):
        return attrs.DIRNAME

    @property
    def fname(self):
        return attrs.FREQNAME

    @property
    def darr(self):
        # Define polar coordinates
        _darr = self._polar_dir(self._darr)
        # Overwrite attributes for labels
        _darr = self._set_labels(_darr)
        # Normalise
        if self.normalised:
            _darr = (_darr - _darr.min()) / (_darr.max() - _darr.min())
        # Set lowest value for masking
        if self.efth_min is not None:
            _darr = _darr.where(_darr >= self.efth_min, self.efth_min)
        # Convert frequencis to periods
        if self.as_period:
            _darr = self._to_period(_darr)
        # Set log10 radii
        if self.logradius:
            _darr = self._to_logradius(_darr)
        return _darr

    @property
    def rmin(self):
        """The radius centre."""
        if self._rmin is None:
            return float(self.darr[self.fname].min())
        if self.logradius and self._rmin > 0:
            return np.log10(self._rmin * LOG_FACTOR)
        else:
            return self._rmin

    @property
    def rmax(self):
        """The radius edge."""
        if self._rmax is None:
            return float(self.darr[self.fname].max())
        if self.logradius and self._rmax > 0:
            return np.log10(self._rmax * LOG_FACTOR)
        else:
            return self._rmax

    @property
    def cbar_ticks(self):
        """Colorbar ticks."""
        if self._cbar_ticks is None and self.normalised and "contour" in self.kind and "levels" not in self._kwargs:
            self._cbar_ticks = CBAR_TICKS
        return self._cbar_ticks

    @property
    def radii_ticks(self):
        """Tick locations for the radii axis."""
        radii_ticks = self._radii_ticks
        # Assign default values
        if self._radii_ticks is None:
            if self.logradius:
                if self.as_period:
                    radii_ticks = RADII_PER_TICKS_LOG
                else:
                    radii_ticks = RADII_FREQ_TICKS_LOG
            else:
                if self.as_period:
                    radii_ticks = RADII_PER_TICKS_LIN
                else:
                    radii_ticks = RADII_FREQ_TICKS_LIN
        # Ensure numpy array
        radii_ticks = np.array(radii_ticks)
        # Taking the log10 if logradius is True
        if self.logradius:
            radii_ticks = np.log10(radii_ticks * LOG_FACTOR)
        # Raise ValueError if radii ticks are not within (rmin, rmax)
        if self.rmin >= radii_ticks.max() or self.rmax <= radii_ticks.min():
            if self.logradius:
                ticks = 10 ** radii_ticks / LOG_FACTOR
                rmin = 10 ** self.rmin / LOG_FACTOR
                rmax = 10 ** self.rmax / LOG_FACTOR
            else:
                ticks = radii_ticks
                rmin = self.rmin
                rmax = self.rmax
            raise ValueError(
                f"radii_ticks '{ticks}' outside the bounds defined by 'rmin={rmin}', "
                f"'rmax={rmax}', perhaps you are trying to define frequency radii "
                "ticks on period radii or vice-versa?"
            )
        # Clipping to radii limits
        radii_ticks = np.unique(radii_ticks.clip(self.rmin, self.rmax))
        return radii_ticks

    @property
    def radii_ticklabels(self):
        """Tick labels for the radii axis."""
        units = self.darr[self.fname].attrs.get("units", "Hz")
        if self.logradius:
            ticks = 10 ** self.radii_ticks / 1000
        else:
            ticks = self.radii_ticks
        ticklabels = [f"{v:g}" for v in ticks]
        ticklabels[-1] = ticklabels[-1] + units
        return ticklabels

    @property
    def kwargs(self):
        _kwargs = {**self._kwargs, **{"x": self.dname, "y": self.fname}}
        if "colors" not in self._kwargs:
            _kwargs.update({"cmap": self.cmap})
        if "contour" in self.kind:
            _kwargs.update({"extend": self.extend})
        if self.normalised and "contour" in self.kind and "levels" not in _kwargs:
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
                closure = closure.assign_coords(
                    {self.dname: darray[self.dname][-1] + dd}
                )
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

    def _to_logradius(self, darr):
        fattrs = darr[self.fname].attrs
        dattrs = darr[self.dname].attrs
        sattrs = darr.attrs
        freqs = np.log10(darr.freq * LOG_FACTOR)
        freqs = freqs.where(np.isfinite(freqs), 0)
        darr = darr.assign_coords({self.fname: freqs})
        darr.attrs = sattrs
        darr[self.fname].attrs = fattrs
        darr[self.dname].attrs = dattrs
        return darr


def polar_plot(*args, **kargs):
    """Plot spectra in polar axis.

    Args:
        - darr (DataArray): Wavespectra DataArray.
        - kind (str): Plot kind, one of (`contourf`, `contour`, `pcolormesh`).
        - normalised (bool): Show efth normalised between 0 and 1.
        - logradius (bool): Set log radii.
        - as_period (bool): Set radii as wave period instead of frequency.
        - rmin (float): Minimum value to clip the radius axis.
        - rmax (float): Maximum value to clip the radius axis.
        - show_theta_labels (bool): Show direction tick labels.
        - show_radii_labels (bool): Show radii tick labels.
        - radii_ticks (array): Tick values for radii.
        - radii_labels_angle (float): Polar angle at which radii labels are positioned.
        - radii_labels_size (float): Fontsize for radii labels.
        - cbar_ticks (array): Tick values for colorbar.
        - cmap (str, obj): Colormap to use.
        - efth_min (float): Clip energy density below this value.
        - kwargs: All extra kwargs are passed to the plotting method defined by `kind`.

    Returns:
        - pobj: The xarray object returned by calling `da.plot.{kind}(**kwargs)`.

    """
    wp = WavePlot(*args, **kargs)
    return wp()
