"""Read Spotter buoy files."""

from xarray.backends import BackendEntrypoint
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import copy
import glob
import getpass
from datetime import datetime, timezone
import json
import numpy as np
import xarray as xr
import pandas as pd

import wavespectra
from wavespectra.core.attributes import set_spec_attributes
from wavespectra.construct.direction import cartwright


SPECTRAL = {
    "varianceDensity": "efth",
    "direction": "dmf",
    "directionalSpread": "dsprf",
    "a1": "a1",
    "b1": "b1",
    "a2": "a2",
    "b2": "b2",
}

PARAMETERS_CSV = {
    "Battery Voltage": "battery",
    "Power": "power",
    "Humidity": "humidity",
    "Significant Wave Height": "hs",
    "Peak Period": "tp",
    "Mean Period": "tm",
    "Peak Direction": "dpm",
    "Peak Directional Spread": "dpspr",
    "Mean Direction": "dm",
    "Mean Directional Spread": "dspr",
    "Latitude": "lat",
    "Longitude": "lon",
    "Wind Speed": "wspd",
    "Wind Direction": "wdir",
    "Surface Temperature": "sst",
}

PARAMETERS_JSON = {
    "significantWaveHeight": "hs",
    "peakPeriod": "tp",
    "meanPeriod": "tm",
    "peakDirection": "dpm",
    "peakDirectionalSpread": "dpspr",
    "meanDirection": "dm",
    "meanDirectionalSpread": "dspr",
    "latitude": "lat",
    "longitude": "lon",
    "windSpeed": "wspd",
    "windDirection": "wdir",
    "surfaceTemperature": "sst",
    "timestamp": "time",
}

METADATA = {
    "battery": {"standard_name": "battery_voltage", "units": "V"},
    "power": {"standard_name": "power", "units": "W"},
    "humidity": {"standard_name": "relative_humidity", "units": "%"},
    "sst": {"standard_name": "sea_surface_temperature", "units": "degC"},
}


def _read_spotter_csv(filename, dd=5.0) -> xr.Dataset:
    """Read Spectra from Spotter CSV file.

    Args:
        - filename (list, str): File name or file glob specifying spotter files to read.
        - dd (float): Directional spacing if 2D spectra are desired.

    """
    spot = SpotterCSV(filename)
    dset = spot.read(dd=dd)
    return dset


def _read_spotter_json(filename, dd=5.0) -> xr.Dataset:
    """Read Spectra from Spotter JSON file.

    Args:
        - filename (list, str): File name or file glob specifying spotter files to read.
        - dd (float): Directional spacing if 2D spectra are desired.

    """
    spot = SpotterJson(filename)
    dset = spot.read(dd=dd)
    return dset


def read_spotter(filename_or_fileglob, filetype=None, dd=5.0) -> xr.Dataset:
    """Read Spectra from Spotter file.

    Args:
        - filename_or_fileglob (list, str): File name or file glob specifying spotter
          files to read.
        - filetype (str): 'json' or 'csv', if not passed inferred from filename.
        - dd (float): Directional spacing if 2D spectra are desired, use None to read
          1D spectra.
    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """
    # Ensure a list of files
    filenames = copy.deepcopy(filename_or_fileglob)
    if isinstance(filename_or_fileglob, list):
        filenames = filename_or_fileglob
    elif Path(filenames).is_file():
        filenames = [filename_or_fileglob]
    else:
        filenames = sorted(glob.glob(filename_or_fileglob))
        if not filenames:
            raise ValueError(f"No files found for '{filename_or_fileglob}'")

    # Infer filetype from filename
    if filetype is None:
        filetype = Path(filenames[0]).suffix.lower().removeprefix(".")
    else:
        filetype = filetype.lower()
    if filetype not in ["json", "csv"]:
        raise ValueError(f"filetype='{filetype}', must be either 'json' or 'csv' ")

    # Read files
    reader = globals()[f"_read_spotter_{filetype}"]
    dslist = []
    for filename in filenames:
        dslist.append(reader(filename, dd))
    return xr.concat(dslist, dim="time")


class Spotter(ABC):
    """Base class for reading spotter files."""

    def __init__(self, filename: str):
        """Read Spectra from Spotter CSV file.

        Args:
            - filename (str): File name specifying spotter file to read.

        """
        self.filename = filename

    @abstractmethod
    def read_spectra(self) -> xr.Dataset:
        """Read spectral data"""
        pass

    @abstractmethod
    def read_params(self) -> xr.Dataset:
        """Read bulk parameters."""
        pass

    def _set_attributes(self, dset: xr.Dataset) -> xr.Dataset:
        set_spec_attributes(dset)
        dset.attrs = {
            "title": f"Spotter buoy spectral data from {Path(self.filename).name}",
            "source": "Spotter wave buoy",
            "history": f"Generated by wavespectra v{wavespectra.__version__}",
            "date_created": f"{datetime.now(timezone.utc)}",
            "creator_name": getpass.getuser(),
            "references": "https://content.sofarocean.com/hubfs/Technical_Reference_Manual.pdf",
        }
        # Extra non-waves attributes
        for key, value in METADATA.items():
            if key in dset:
                dset[key].attrs = value
        return dset

    def read(self, dd: float = None) -> xr.Dataset:
        """Read spotter file as wavespectra dataset.

        Args:
            - dd (float): Directional spacing if 2D spectra are desired.

        Returns:
            - dset (SpecDataset): wavespectra dataset object.

        """
        dset = xr.merge([self.read_spectra(), self.read_params()]).sortby("time")
        if dd is not None:
            dir = np.arange(0, 360, dd)
            dir = xr.DataArray(dir, coords=dict(dir=dir), name="dir")
            cos2 = cartwright(dir=dir, dm=dset.dmf, dspr=dset.dsprf)
            dset["efth"] = dset.efth * cos2
        return self._set_attributes(dset)


class SpotterCSV(Spotter):
    """Read Spectra from Spotter Json file."""

    @cached_property
    def data(self) -> pd.DataFrame:
        """The data content in the csv file."""
        data = pd.read_csv(self.filename)
        data.columns = [c.split("(")[0].strip() for c in data.columns]
        return data

    @cached_property
    def time(self) -> xr.DataArray:
        """Time coord."""
        data = pd.to_datetime(self.data["Epoch Time"], unit="s")
        return xr.DataArray(data, coords=dict(time=data), name="time")

    @cached_property
    def freq(self) -> xr.DataArray:
        """Frequency coord."""
        data = self.data.filter(regex=r"^f_\d+").drop_duplicates()
        if data.shape[0] > 1:
            raise NotImplementedError("Varying frequency arrays not yet supported")
        data = data.values[0]
        return xr.DataArray(data, coords=dict(freq=data), name="freq")

    def read_params(self) -> xr.Dataset:
        """Read bulk parameters."""
        pattern = "|".join(f"^{prefix}" for prefix in PARAMETERS_CSV)
        data = self.data.filter(regex=pattern).rename(columns=PARAMETERS_CSV)
        return data.set_index(self.time.to_series()).to_xarray()

    def read_spectra(self) -> xr.Dataset:
        """Read spectral data"""
        coords = {"time": self.time, "freq": self.freq}
        dset = xr.Dataset()
        for col, var in SPECTRAL.items():
            dset[var] = xr.DataArray(self.data.filter(regex=f"{col}_"), coords=coords)
        return dset


class SpotterJson(Spotter):
    """Read Spectra from Spotter Json file."""

    @cached_property
    def data(self) -> dict:
        """The data content in the json file."""
        with open(self.filename) as json_file:
            return json.load(json_file)["data"]

    @cached_property
    def time(self) -> xr.DataArray:
        """Time coord."""
        data = pd.DataFrame(self.data["waves"], columns=["timestamp"])
        data = pd.to_datetime(data.timestamp).dt.tz_localize(None)
        return xr.DataArray(data, coords=dict(time=data), name="time")

    @cached_property
    def freq(self) -> xr.DataArray:
        """Frequency coord."""
        data = pd.DataFrame(self.data["frequencyData"], columns=["frequency"])
        data = data.drop_duplicates()
        if data.shape[0] > 1:
            raise NotImplementedError("Varying frequency arrays not yet supported")
        data = data.values[0][0]
        return xr.DataArray(data, coords=dict(freq=data), name="freq")

    def read_params(self) -> xr.Dataset:
        """Read bulk parameters."""
        data = pd.DataFrame(self.data["waves"]).rename(columns=PARAMETERS_JSON)
        return data.drop("time", axis=1).set_index(self.time.to_series()).to_xarray()

    def read_spectra(self) -> xr.Dataset:
        """Read spectral data"""
        data = pd.DataFrame(self.data["frequencyData"])
        coords = {"time": self.time, "freq": self.freq}
        dset = xr.Dataset()
        for var in SPECTRAL.keys():
            dset[var] = xr.DataArray(np.stack(data[var].values), coords=coords)
        return dset.rename(SPECTRAL)


class SpotterBackendEntrypoint(BackendEntrypoint):
    """Spotter backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        filetype=None,
    ):
        return read_spotter(filename_or_obj, filetype=filetype)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open Spotter spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
