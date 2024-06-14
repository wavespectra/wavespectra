"""WaveSpectra reader for Obscape CSV files (as downloaded from portal).

Buoy data: https://obscape.com/site/
Note: for raw data use the python file that can be downloaded from the obs site.

Revision history:
----------------
2024-03-11 : First version



"""

import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import xarray as xr

from wavespectra.core.attributes import set_spec_attributes


logger = logging.getLogger(__name__)


def _read_obscape_file(filename: str) -> dict:
    """Read an Obscape file.

    args:
        - filename (str): The filename to read.

    returns:
        - R (dict): A dictionary containing the data read from the file.

    Notes:
        - metadata is marked with a # symbol.
        - time is in filename, UTC.
        - all other lines are a CSV file.

    """
    filename = Path(filename)
    assert filename.exists()

    info = []

    with open(filename, "r") as f:
        lines = f.readlines()

        metadata = dict()

        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=")
                    metadata[key[1:].strip()] = value.strip()
                else:
                    info.append(line)

    # read the csv data using numpy, skipping lines that start with #
    data = np.genfromtxt(filename, delimiter=",", comments="#")

    # get frequencies
    freq = metadata["Rows [Hz]"]
    freq = freq.split(",")
    freq = [float(f) for f in freq]

    dir = metadata["Columns [deg]"]

    # split on , or space
    dir = dir.replace(" ", ",").split(",")

    # check if it makes sense
    dd = float(dir[1]) - float(dir[0])

    dirs = np.arange(0, 360, dd)

    # check dims

    if data.shape[0] != len(freq):
        logger.warning(f"Frequency dimension mismatch: {data.shape[0]} != {len(freq)}")

    if data.shape[1] != len(dirs):
        logger.warning(f"Direction dimension mismatch: {data.shape[1]} != {len(dirs)}")

    metadata["info"] = info

    # return as a dict
    R = dict()
    R["freq"] = freq
    R["dir"] = dirs
    R["data"] = data

    timestamp = metadata["Timestamp"]  # unix timestamp

    # convert timestamp to datetime object
    R["utc"] = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).replace(
        tzinfo=None
    )

    R["metadata"] = metadata

    return R


def read_obscape(filename_or_fileglob):
    """Read spectra from Obscape wave buoy csv files.

    The CSV files are downloaded from the Obscape portal.

    Args:
        - filename_or_fileglob (str, list): A single filename or a list of filenames
          or a fileglob pattern.

    Returns:
        - dset (SpecDataset): A wavespectra SpecDataset object.

    """

    # step 1: get the files

    if isinstance(filename_or_fileglob, list):
        files = filename_or_fileglob
    else:
        path = Path(filename_or_fileglob)
        files = sorted(path.absolute().parent.glob(path.name))

    if not files:
        raise ValueError(f"No files found from {filename_or_fileglob}")

    # step 2: get the data

    R = []
    for file in files:
        R.append(_read_obscape_file(file))

    # step 3: construct the data

    metadata = R[0]["metadata"]  # use the first read spectrum
    dirs = R[0]["dir"]
    freqs = R[0]["freq"]

    efth = [d["data"] for d in R]
    times = [d["utc"] for d in R]

    ds = xr.DataArray(
        data=efth,
        coords={"time": times, "freq": freqs, "dir": dirs},
        dims=("time", "freq", "dir"),
        name="efth",
    ).to_dataset()

    ds = ds.sortby("time", ascending=True)

    # scale
    ds["efth"] = ds["efth"] * np.pi / 180  # convert to m2/Hz/deg

    for key in [
        "Station name",
        "Device type",
        "Device serial",
        "Latitude [deg]",
        "Longitude [deg]",
        "Magnetic declination (corrected) [deg]",
        "Directions",
        "info",
    ]:
        try:
            ds.attrs[key] = metadata[key]
        except KeyError:
            pass

    # Set attributes
    set_spec_attributes(ds)

    # add site dimension
    ds["site"] = [0]

    return ds


def _get_timestamp(stem):

    try:
        year = int(stem[:4])
        month = int(stem[4:6])
        day = int(stem[6:8])
        hour = int(stem[9:11])
        minute = int(stem[11:13])
        second = int(stem[13:15])
    except ValueError:
        logger.warning(
            "Filename does not contain a valid UTC timestamp, expect the filename "
            f"to start with yyyymmdd_hhmmss but got: {stem}"
        )
        return None

    # make a python datetime object
    utc = datetime(year, month, day, hour, minute, second)

    return utc


def _get_obs_files(directory, start_date=None, end_date=None):
    """Get a list of csv files in a directory with timestamps in a given range.

    This function return all the .csv files in the directory that have a timestamp
    greater than or equal to start_date and less than or equal to end_date. Timestamps
    are extracted from the filename which are expected to be in the format:

    `yyyymmdd_hhmmss.....csv`

    Args:
        - directory (str): The directory containing the Obscape files.
        - start_date (datetime): The start date to filter the files.
        - end_date (datetime): The end date to filter the files.

    Returns:
        - R (list): A list of Path objects.

    """

    directory = Path(directory)

    assert directory.exists()

    files = directory.glob("*.csv")
    R = []
    for file in files:

        timestamp = _get_timestamp(file.stem)

        if start_date is not None:
            if timestamp < start_date:
                continue

        if end_date is not None:
            if timestamp > end_date:
                continue

        R.append(file)

    return R


def read_obscape_dir(directory, start_date=None, end_date=None):
    """Read obscape spectra files from directory.

    This function reads all the files in the directory that have a timestamp greater
    than or equal to `start_date` and less than or equal to `end_date`. Timestamps are
    extracted from the filename. The filename is expected to start with the timestamp
    which is expected to be in the format yyyymmdd_hhmmss, e.g.,
    `20240214_000000_wavebuoy_xxx_spec2D.csv`.

    Args:
        - directory (str): The directory containing the Obscape files.
        - start_date (datetime): The start date to filter the files.
        - end_date (datetime): The end date to filter the files.

    Returns:
        - dset (SpecDataset): A wavespectra SpecDataset object.

    """

    files = _get_obs_files(directory, start_date, end_date)

    return read_obscape(files)
