"""Read ERA5 2D Wave Spectra NetCDF files"""
from xarray.backends import BackendEntrypoint
import json
import datetime
import xarray as xr

from wavespectra.core.attributes import set_spec_attributes


def read_json(filename_or_obj, date_format="%Y-%m-%dT%H:%M:%SZ"):
    """Read Spectra from json.

    The wavespectra json format is produced from `SpecDataset.to_json` by running
    `Dataset.to_dict` and converting times into iso8601 strings.

    Args:
        - filename_or_obj (str): filename or filelike object of json to read.
        - date_format(str): strftime format for de-serializing datetimes.

    Returns:
        - dset (SpecDataset): spectra dataset object read from json file.

    """
    try:
        dset_dict = json.load(filename_or_obj)
    except AttributeError:
        with open(filename_or_obj) as fp:
            dset_dict = json.load(fp)

    for item in ["coords", "data_vars"]:
        if "time" in dset_dict[item]:
            time_strings = dset_dict[item]["time"]["data"]
            times = [datetime.datetime.strptime(t, date_format) for t in time_strings]
            dset_dict[item]["time"]["data"] = times

    dset = xr.Dataset.from_dict(dset_dict)
    set_spec_attributes(dset)
    return dset


class JsonBackendEntrypoint(BackendEntrypoint):
    """Jason backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        return read_json(filename_or_obj)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open Json spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
