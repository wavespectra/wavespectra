"""Read SWAN spectra files.

Functions:
    read_spotter: Read Spectra from Spotter JSON file
    read_spotters: Read multiple spotter files into single Dataset

"""
import glob
import os
import getpass
import datetime
import json
import numpy as np
import xarray as xr
import pandas as pd
from dateutil.parser import parse

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes


def read_spotter(filename, filetype=None):
    """Read Spectra from Spotter file.

    Args:
        - filename (list, str): File name or file glob specifying spotter files to read.
        - filetype (str): 'json' or 'csv', if not passed inferred from filename.

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """

    if filename is not isinstance(filename, list):
        filename = [filename]

    if filetype is None:
        filetype = os.path.splitext(filename[0])[1]

    filetype = filetype.removeprefix(".").lower()

    if filetype == "json":
        return _read_spotter_json(filename)
    elif filetype == "csv":
        dslist = []
        for fn in filename:
            dslist.append(_read_spotter_csv(fn))
        return xr.concat(dslist, dim="time")
    else:
        raise ValueError(f"filetype='{filetype}', must be either 'json' or 'csv' ")


def _read_spotter_csv(filename):
    """Read Spectra from Spotter CSV file.

    Args:
        - filename (str): File name specifying spotter file to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """

    # Read bulk parameters
    dat_bulk = pd.read_csv(
        filename,
        index_col=3,
        usecols=np.insert(np.arange(13), -1, [364, 365, 366]),
    )
    dat_bulk.index = pd.to_datetime(dat_bulk.index, unit="s")
    dat_bulk.columns = dat_bulk.columns.str.strip()
    dat_bulk.index.name = dat_bulk.index.name.strip()
    b = dat_bulk.to_xarray()

    # Frequency array
    dat_f = pd.read_csv(
        filename,
        index_col=[],
        usecols=np.arange(13, 13 + 38 + 1),
    )
    dat_f.columns = dat_f.columns.str.strip()
    freq_array = dat_f.iloc[0].to_xarray()
    freq_array.name = "Frequency"

    # Frequency spacing
    dat_df = pd.read_csv(
        filename,
        index_col=[],
        usecols=np.arange(13 + 38 + 1, 13 + 2 * (38 + 1)),
    )
    dat_df.columns = dat_df.columns.str.strip()
    df_array = (
        dat_df.iloc[0]
        .to_xarray()
        .assign_coords(dict(index=freq_array.values))
        .rename(dict(index="Frequency"))
    )
    df_array.name = "df"
    df_array.attrs["long_name"] = "Frequency spacing"
    df_array.attrs["units"] = "Hz"

    # a and b parameters
    names = ["a1", "b1", "a2", "b2"]
    tmp_list = []
    for idx, name in enumerate(names):
        dat_tmp = pd.read_csv(
            filename,
            index_col=[0],
            usecols=np.insert(
                np.arange(13 + (2 + idx) * (38 + 1), 13 + (3 + idx) * (38 + 1)), 0, 3
            ),
        )
        dat_tmp.index = pd.to_datetime(dat_tmp.index, unit="s")
        dat_tmp.columns = dat_tmp.columns.str.strip()
        dat_tmp.index.name = dat_tmp.index.name.strip()
        tmp_da = dat_tmp.to_xarray().to_array(dim="Frequency", name=name)
        tmp_da = tmp_da.assign_coords({"Frequency": freq_array.values})
        tmp_list.append(tmp_da)

    ab_ds = xr.merge(tmp_list)

    # Spectral density, spreading, etc.
    dat_S = pd.read_csv(
        filename,
        index_col=[0],
        usecols=np.insert(np.arange(13 + 6 * (38 + 1), 13 + 7 * (38 + 1)), 0, 3),
    )
    dat_S.index = pd.to_datetime(dat_S.index, unit="s")
    dat_S.columns = dat_S.columns.str.strip()
    dat_S.index.name = dat_S.index.name.strip()
    S = dat_S.to_xarray().to_array(dim="Frequency", name="Variance density")
    S = S.assign_coords({"Frequency": freq_array.values})

    dat_dir = pd.read_csv(
        filename,
        index_col=[0],
        usecols=np.insert(np.arange(13 + 7 * (38 + 1), 13 + 8 * (38 + 1)), 0, 3),
    )
    dat_dir.index = pd.to_datetime(dat_dir.index, unit="s")
    dat_dir.columns = dat_dir.columns.str.strip()
    dat_dir.index.name = dat_dir.index.name.strip()
    Dir = dat_dir.to_xarray().to_array(dim="Frequency", name="Direction")
    Dir = Dir.assign_coords({"Frequency": freq_array.values})

    dat_spread = pd.read_csv(
        filename,
        index_col=[0],
        usecols=np.insert(np.arange(13 + 8 * (38 + 1), 13 + 9 * (38 + 1)), 0, 3),
    )
    dat_spread.index = pd.to_datetime(dat_spread.index, unit="s")
    dat_spread.index.name = dat_spread.index.name.strip()
    dat_spread.columns = dat_spread.columns.str.strip()
    spread = dat_spread.to_xarray().to_array(dim="Frequency", name="Directional spread")
    spread = spread.assign_coords({"Frequency": freq_array.values})

    ds = xr.merge([b, ab_ds, S, Dir, spread, df_array])

    ds["Epoch Time"].attrs["long_name"] = "Epoch time"

    ds["Battery Voltage (V)"].attrs["units"] = "V"
    ds["Battery Voltage (V)"].attrs["long_name"] = "Battery voltage"

    ds["Power (W)"].attrs["units"] = "W"
    ds["Power (W)"].attrs["long_name"] = "Battery power"

    ds["Humidity (%rel)"].attrs["units"] = "1"
    ds["Humidity (%rel)"].attrs["standard_name"] = "relative_humidity"
    ds["Humidity (%rel)"].attrs["long_name"] = "Relative humidity"

    ds["Significant Wave Height (m)"].attrs["units"] = "m"
    ds["Significant Wave Height (m)"].attrs[
        "standard_name"
    ] = "sea_surface_wave_significant_height"
    ds["Significant Wave Height (m)"].attrs["long_name"] = "Significant wave height"

    ds["Direction"].attrs["units"] = "degree"
    ds["Direction"].attrs["long_name"] = ""

    ds["Peak Period (s)"].attrs["units"] = "s"
    ds["Peak Period (s)"].attrs[
        "standard_name"
    ] = "sea_surface_wave_period_at_variance_spectral_density_maximum"
    ds["Peak Period (s)"].attrs["long_name"] = "Peak period"

    ds["Mean Period (s)"].attrs["units"] = "s"
    ds["Mean Period (s)"].attrs[
        "standard_name"
    ] = "sea_surface_wave_zero_upcrossing_period"
    ds["Mean Period (s)"].attrs["long_name"] = "Mean period"

    ds["Peak Direction (deg)"].attrs["units"] = "degree"
    ds["Peak Direction (deg)"].attrs[
        "standard_name"
    ] = "sea_surface_wave_from_direction_at_variance_spectral_density_maximum"
    ds["Peak Direction (deg)"].attrs["long_name"] = "Peak direction"

    ds["Peak Directional Spread (deg)"].attrs["units"] = "degree"
    ds["Peak Directional Spread (deg)"].attrs[
        "standard_name"
    ] = "sea_surface_wave_directional_spread_at_variance_spectral_density_maximum"
    ds["Peak Directional Spread (deg)"].attrs["long_name"] = "Peak directional spread"

    ds["Mean Direction (deg)"].attrs["units"] = "degree"
    ds["Mean Direction (deg)"].attrs[
        "standard_name"
    ] = "sea_surface_wave_from_direction"
    ds["Mean Direction (deg)"].attrs["long_name"] = "Mean direction"

    ds["Mean Directional Spread (deg)"].attrs["units"] = "degree"
    ds["Mean Directional Spread (deg)"].attrs["long_name"] = "Mean directional spread"

    ds["Latitude (deg)"].attrs["units"] = "degree_north"
    ds["Latitude (deg)"].attrs["standard_name"] = "latitude"
    ds["Latitude (deg)"].attrs["long_name"] = "Latitude"

    ds["Longitude (deg)"].attrs["units"] = "degree_east"
    ds["Longitude (deg)"].attrs["standard_name"] = "longitude"
    ds["Longitude (deg)"].attrs["long_name"] = "Longitude"

    ds["Wind Speed (m/s)"].attrs["units"] = "m/s"
    ds["Wind Speed (m/s)"].attrs["standard_name"] = "wind_speed"
    ds["Wind Speed (m/s)"].attrs["long_name"] = "Wind speed"

    ds["Wind Direction (deg)"].attrs["units"] = "degree"
    ds["Wind Direction (deg)"].attrs["standard_name"] = "wind_from_direction"
    ds["Wind Direction (deg)"].attrs["long_name"] = "Wind direction"

    ds["Surface Temperature (°C)"] = 274.15 * ds["Surface Temperature (°C)"]
    ds["Surface Temperature (°C)"].attrs["units"] = "K"
    ds["Surface Temperature (°C)"].attrs["standard_name"] = "sea_surface_temperature"
    ds["Surface Temperature (°C)"].attrs["long_name"] = "Surface temperature"

    ds["Frequency"].attrs["units"] = "Hz"
    ds["Frequency"].attrs["standard_name"] = "wave_frequency"
    ds["Frequency"].attrs["long_name"] = "Frequency"

    ds["Variance density"].attrs["units"] = "m2 s"
    ds["Variance density"].attrs[
        "standard_name"
    ] = "sea_surface_wave_variance_spectral_density"
    ds["Variance density"].attrs["long_name"] = "Spectral density"

    ds["Directional spread"].attrs["units"] = "degree"
    ds["Directional spread"].attrs[
        "standard_name"
    ] = "sea_surface_wave_directional_spread"
    ds["Directional spread"].attrs["long_name"] = "Directional spreading"

    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["source"] = f"Sofar Spotter buoy, {filename}"
    ds.attrs[
        "history"
    ] = f"generated {datetime.datetime.now().strftime('%Y-%m-%d @ %H:%M:%S')} by {getpass.getuser()}"
    ds.attrs[
        "references"
    ] = "https://content.sofarocean.com/hubfs/Technical_Reference_Manual.pdf"

    ds = ds.rename(
        {
            "Epoch Time": "time",
            "Frequency": "freq",
            "Battery Voltage (V)": "batter_voltage",
            "Variance density": "efth",
            "Direction": "dir",
            "Power (W)": "battery_power",
            "Humidity (%rel)": "humidity",
            "Significant Wave Height (m)": "Hm0",
            "Peak Period (s)": "Tp",
            "Mean Period (s)": "Tm",
            "Peak Direction (deg)": "peak_dir",
            "Peak Directional Spread (deg)": "peak_spread",
            "Mean Direction (deg)": "mean_dir",
            "Mean Directional Spread (deg)": "mean_spread",
            "Directional spread": "spread",
            "Latitude (deg)": "lat",
            "Longitude (deg)": "lon",
            "Wind Speed (m/s)": "wspd",
            "Wind Direction (deg)": "wdir",
            "Surface Temperature (°C)": "temperature",
        }
    )

    ds = ds.sortby("time")

    return ds


def _read_spotter_json(filename):
    """Read Spectra from Spotter JSON file.

    Args:
        - filename (list, str): File name or file glob specifying spotter files to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """
    spot = Spotter(filename)
    dset = spot.run()
    return dset


class Spotter:
    def __init__(self, filename_or_fileglob, toff=0):
        """Read wave spectra file from TRIAXYS buoy.

        Args:
            - filename_or_fileglob (str, list): filename or fileglob
              specifying files to read.
            - toff (float): time offset in hours to account for
              time zone differences.

        Returns:
            - dset (SpecDataset) wavespectra SpecDataset instance.

        Remark:
            - frequencies and directions from first file are used as reference
              to interpolate spectra from other files in case they differ.
              In fact interpolation is still not implemented here, code will break
              if spectral coordinates are different.

        """
        self._filename_or_fileglob = filename_or_fileglob
        self.toff = toff

    def run(self):
        """Returns wave spectra dataset from one or more spotter files."""
        dsets = []
        for self.filename in self.filenames:
            self._load_json()
            self._set_arrays_from_json()
            dsets.append(self._construct_dataset())
        # Ensure same spectral coords across files, interp needs to be implemented
        if not self._is_unique([dset.freq.values for dset in dsets]):
            raise NotImplementedError(
                "Varying frequency arrays between spotter files not yet supported."
            )
        if not self._is_unique([dset.dir.values for dset in dsets]):
            raise NotImplementedError(
                "Varying direction arrays between spotter files not yet supported."
            )
        # Concatenating datasets from multiple files
        self.dset = xr.concat(dsets, dim="time")
        return self.dset

    def _is_unique(self, arrays):
        """Returns True if all iterators in arrays are the same."""
        if len(set(tuple(array) for array in arrays)) == 1:
            return True
        else:
            return False

    def _set_arrays_from_json(self):
        """Set spectra attributes from arrays in json blob."""
        # Spectra
        keys = self.data["data"]["frequencyData"][0].keys()
        for key in keys:
            setattr(
                self,
                key,
                [sample[key] for sample in self.data["data"]["frequencyData"]],
            )
        # Keep here only for checking - timestamps seem to differ
        self.timestamp_spec = self.timestamp
        self.latitude_spec = self.latitude
        self.longitude_spec = self.longitude
        # Parameters
        if "waves" in self.data["data"]:
            keys = self.data["data"]["waves"][0].keys()
            for key in keys:
                setattr(
                    self, key, [sample[key] for sample in self.data["data"]["waves"]]
                )
        # Keep here only for checking - timestamps seem to differ
        self.timestamp_param = self.timestamp
        self.latitude_param = self.latitude
        self.longitude_param = self.longitude

    def _construct_dataset(self):
        """Construct wavespectra dataset."""
        self.dset = xr.DataArray(
            data=self.efth, coords=self.coords, dims=self.dims, name=attrs.SPECNAME
        ).to_dataset()
        self.dset[attrs.LATNAME] = xr.DataArray(
            data=self.latitude, coords={"time": self.dset.time}, dims=("time")
        )
        self.dset[attrs.LONNAME] = xr.DataArray(
            data=self.longitude, coords={"time": self.dset.time}, dims=("time")
        )
        set_spec_attributes(self.dset)
        return self.dset

    def _load_json(self):
        """Load data from json blob."""
        with open(self.filename) as json_file:
            self.data = json.load(json_file)
        try:
            self.data["data"]["spotterId"]
        except KeyError:
            raise OSError(f"Not a Spotter Spectra file: {self.filename}")

    @property
    def time(self):
        """The time coordinate values."""
        return [
            parse(time).replace(tzinfo=None) - datetime.timedelta(hours=self.toff)
            for time in self.timestamp
        ]

    @property
    def efth(self):
        """The Variance density data values."""
        return [np.expand_dims(varden, axis=1) for varden in self.varianceDensity]

    @property
    def freq(self):
        """The frequency coordinate values."""
        if not self._is_unique(self.frequency):
            raise NotImplementedError(
                "Varying frequency arrays in single file not yet supported."
            )
        return self.frequency[0]

    @property
    def dir(self):
        """The direction coordinate values, currently set to [0.] for 1D spectra."""
        return [0.0]

    @property
    def dims(self):
        """The dataset dimensions."""
        return (attrs.TIMENAME, attrs.FREQNAME, attrs.DIRNAME)

    @property
    def coords(self):
        """The dataset coordinates."""
        return {
            attrs.TIMENAME: self.time,
            attrs.FREQNAME: self.freq,
            attrs.DIRNAME: self.dir,
        }

    @property
    def filenames(self):
        if isinstance(self._filename_or_fileglob, list):
            filenames = sorted(self._filename_or_fileglob)
        elif isinstance(self._filename_or_fileglob, str):
            filenames = sorted(glob.glob(self._filename_or_fileglob))
        if not filenames:
            raise ValueError(f"No file located in {self._filename_or_fileglob}")
        return filenames
