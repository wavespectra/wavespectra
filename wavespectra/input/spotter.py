"""Read SWAN spectra files.

Functions:
    read_spotter: Read Spectra from Spotter JSON file
    read_spotters: Read multiple spotter files into single Dataset

"""
import glob
import datetime
import json
import numpy as np
import xarray as xr
from dateutil.parser import parse

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes


def read_spotter(filename):
    """Read Spectra from spotter JSON file.

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
