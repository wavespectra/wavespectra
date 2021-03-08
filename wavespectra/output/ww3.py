"""Native WAVEWATCH3 output plugin."""
import os
import yaml
import numpy as np
import xarray as xr

# from wavespectra import __version__
from wavespectra.core.attributes import attrs
from wavespectra.core.utils import R2D


MAPPING = {
    "time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "station": attrs.SITENAME,
    "efth": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "wnddir": attrs.WDIRNAME,
    "wnd": attrs.WSPDNAME,
}

VAR_ATTRIBUTES = yaml.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ww3.yml")),
    Loader=yaml.Loader,
)
TIME_UNITS = VAR_ATTRIBUTES["time"].pop("units")


def to_ww3(self, filename, ncformat="NETCDF4", compress=False):
    """Save spectra in native WW3 netCDF format.

    Args:
        - filename (str): name of output WW3 netcdf file.
        - ncformat (str): netcdf format for output, see options in native
          to_netcdf method.
        - compress (bool): if True output is compressed, has no effect for
          NETCDF3.

    """
    other = self.copy(deep=True)
    # Expanding lon/lat dimensions
    other[attrs.LONNAME] = other[attrs.LONNAME].expand_dims(
        {attrs.TIMENAME: other[attrs.TIMENAME]}
    )
    other[attrs.LATNAME] = other[attrs.LATNAME].expand_dims(
        {attrs.TIMENAME: other[attrs.TIMENAME]}
    )
    # Converting to radians
    other[attrs.SPECNAME] *= R2D
    # frequency bounds
    df = np.hstack((0, np.diff(other[attrs.FREQNAME]) / 2))
    other["frequency1"] = other[attrs.FREQNAME] - df
    df = np.hstack((np.diff(other[attrs.FREQNAME]) / 2, 0))
    other["frequency2"] = other[attrs.FREQNAME] + df
    # Direction in going-to convention
    other[attrs.DIRNAME] = (other[attrs.DIRNAME] + 180) % 360
    # station_name variable
    arr = np.array(
        [[c for c in f"{s:06.0f}"] + [""] * 10 for s in other.site.values], dtype="|S1"
    )
    other["station_name"] = xr.DataArray(
        data=arr,
        coords={"site": other.site, "string16": [np.nan for i in range(16)]},
        dims=("site", "string16"),
    )
    # Renaming
    mapping = {v: k for k, v in MAPPING.items() if v in self.variables}
    other = other.rename(mapping)
    # Setting attributes
    other.attrs.update(VAR_ATTRIBUTES["global"])
    for var_name, var_attrs in VAR_ATTRIBUTES.items():
        if var_name in other:
            other[var_name].attrs = var_attrs
    if "time" in other:
        other.time.encoding["units"] = TIME_UNITS
        times = other.time.to_index().to_pydatetime()
        other.attrs.update(
            {
                "start_date": f"{min(times):%Y-%m-%d %H:%M:%S}",
                "stop_date": f"{max(times):%Y-%m-%d %H:%M:%S}",
            }
        )
        if len(times) > 1:
            hours = round((times[1] - times[0]).total_seconds() / 3600)
            other.attrs.update({"field_type": f"{hours}-hourly"})
    if "latitude" in other.dims:
        other.attrs.update(
            {
                "southernmost_latitude": other.latitude.values.min(),
                "northernmost_latitude": other.latitude.values.max(),
                "latitude_resolution": (other.latitude[1] - other.latitude[0]).values,
                "westernmost_longitude": other.longitude.values.min(),
                "easternmost_longitude": other.longitude.values.max(),
                "longitude_resolution": (
                    other.longitude[1] - other.longitude[0]
                ).values,
            }
        )
    other.attrs.update(
        {
            "product_name": os.path.basename(filename),
            # "format_version": f"wavespectra-{__version__}"
        }
    )
    # Dumping
    other.to_netcdf(filename)
