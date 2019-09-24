"""Read SWAN spectra files.

Functions:
    read_spotter: Read Spectra from Spotter JSON file
    read_spotters: Read multiple spotter files into single Dataset

"""
import copy
import datetime
import glob
import json
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from dateutil import parser

from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.misc import interp_spec
from wavespectra.specdataset import SpecDataset


def test(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def read_spotter(filename, dirorder=True, as_site=None):
    """Read Spectra from spotter JSON file.

    Args:
        - dirorder (bool): If True reorder spectra so that directions are
          sorted.
        - as_site (bool): If True locations are defined by 1D site dimension.

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """

    spot = Spotter(filename)
    spot.run()
    return spot.dset


class Spotter(object):
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
              This is expected for spotter buoys

        """
        self._filename_or_fileglob = filename_or_fileglob
        self.toff = toff
        self.stream = None
        self.is_dir = None
        self.time_list = []
        self.spec_list = []
        self.header_keys = header_keys = [
            "is_triaxys",
            "is_dir",
            "time",
            "nf",
            "f0",
            "df",
            "fmin",
            "fmax",
            "ddir",
        ]

    def run(self):
        for ind, self.filename in enumerate(self.filenames):
            self.load()
            if ind == 0:
                try:
                    self.spotterId = self.data["data"]["spotterId"]
                except Exception as e:
                    raise IOError("Not a Spotter Spectra file.")
                self.interp_freq = copy.deepcopy(self.freqs)
                self.interp_dir = copy.deepcopy(self.dirs)
            self.read_data()
        self.construct_dataset()

    def _append_spectrum(self, time):
        """Append spectra after ensuring same spectral basis."""
        self.spec_list.append(
            interp_spec(
                inspec=self.spec_data,
                infreq=self.freqs,
                indir=self.dirs,
                outfreq=self.interp_freq,
                outdir=self.interp_dir,
            )
        )
        self.time_list.append(time)

    def read_data(self):
        try:
            for ii in range(self.nrecs):
                self.spec_data = np.zeros((len(self.freqs), len(self.dirs)))
                # for i in range(self.nrecs):
                self.spec_data[ii, :] = self.data["data"]["frequencyData"][ii][
                    "varianceDensity"
                ]
                time = parser.parse(self.data["data"]["frequencyData"][ii]["timestamp"])
                self._append_spectrum(time)
        except ValueError as err:
            raise ValueError("Cannot read {}:\n{}".format(self.filename, err))

    def load(self):
        with open(self.filename) as json_file:
            self.data = json.load(json_file)

    def construct_dataset(self):
        self.dset = xr.DataArray(
            data=self.spec_list, coords=self.coords, dims=self.dims, name=attrs.SPECNAME
        ).to_dataset()
        set_spec_attributes(self.dset)
        if not self.is_dir:
            self.dset = self.dset.isel(drop=True, **{attrs.DIRNAME: 0})
            self.dset[attrs.SPECNAME].attrs.update(units="m^{2}.s")

    def open(self):
        self.stream = open(self.filename, "r")

    def close(self):
        if self.stream and not self.stream.closed:
            self.stream.close()

    @property
    def dims(self):
        return (attrs.TIMENAME, attrs.FREQNAME, attrs.DIRNAME)

    @property
    def coords(self):
        _coords = OrderedDict(
            (
                (attrs.TIMENAME, self.time_list),
                (attrs.FREQNAME, self.interp_freq),
                (attrs.DIRNAME, self.interp_dir),
            )
        )
        return _coords

    @property
    def dirs(self):
        return self.data["data"]["frequencyData"][0]["direction"]

    @property
    def freqs(self):
        return self.data["data"]["frequencyData"][0]["frequency"]

    @property
    def nrecs(self):
        return len(self.data["data"]["frequencyData"])

    @property
    def filenames(self):
        if isinstance(self._filename_or_fileglob, list):
            filenames = sorted(self._filename_or_fileglob)
        elif isinstance(self._filename_or_fileglob, str):
            filenames = sorted(glob.glob(self._filename_or_fileglob))
        if not filenames:
            raise ValueError("No file located in {}".format(self._filename_or_fileglob))
        return filenames


d = read_spotter("../../tests/sample_files/spotter_20180214.json")
