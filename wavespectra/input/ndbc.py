import glob
import copy
import datetime
import os
import gzip
from collections import OrderedDict
from dateutil import parser
import numpy as np
import pandas as pd
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes

IPI = 1.0 / np.pi
D2R = np.pi / 180.0


def read_file(filename):
    basename_parts = os.path.basename(filename).split(".")
    compressed = "gzip" if basename_parts[-1] == "gz" else None
    name = basename_parts[0]
    if compressed:
        f = gzip.open(filename)
    else:
        f = open(filename)
    header = f.readline()
    # Look for header like this: #YY  MM DD hh mm Sep_Freq  < spec_1 (freq_1) spec_2 (freq_2) spec_3 (freq_3) ... >
    if header.strip()[-1] == ">":  # Realtime file
        df = pd.read_csv(
            f,
            delimiter="\s+",
            compression=compressed,
            engine="python",
            header=None,
            parse_dates={"time": [0, 1, 2, 3, 4]},
            date_parser=lambda x: datetime.datetime.strptime(x, "%Y %m %d %H %M"),
            index_col=0,
        )
        freqcols = df.select_dtypes(object)  # Get all columns with the frequency
        if not (freqcols.nunique() == 1).all():
            raise IOError("NDBC file has varying frequencies in same file")
        freqs = freqcols.iloc[0].apply(
            lambda x: float(x.lstrip("(").rstrip(")"))
        )  # convert to numeric values
        df.drop(columns=freqcols.columns, inplace=True)
        df.rename(
            columns={c: freqs.get(c + 1, "Sep_Freq") for c in df.columns}, inplace=True
        )
    else:
        f.seek(0, 0)
        df = pd.read_csv(
            f,
            delimiter="\s+",
            engine="python",
            header=[0],
            parse_dates={"time": [0, 1, 2, 3, 4]},
            date_parser=lambda x: datetime.datetime.strptime(x, "%Y %m %d %H %M"),
            index_col=0,
        )
    f.close()
    df.name = name
    return df


def construct_spectra(spden, swdir1, swdir2, swr1, swr2, dirs):
    dirmat = dirs.reshape((1, 1, -1))
    D_fd = (
        IPI
        * (
            0.5
            + swr1 * np.cos(D2R * (dirmat - swdir1))
            + swr2 * np.cos(2 * D2R * (dirmat - swdir2))
        )
        * D2R
    )
    S = spden * D_fd
    return S


def read_ndbc(filename, dirs=np.arange(0, 360, 10)):
    """Read spectra from NDBC wave buoy ASCII files.

    Both the history and realtime formats are supported. Realtime formats are decribed
    at https://www.ndbc.noaa.gov/measdes.shtml.

    Args:
        - filename (str) or filenames (list): filename of 1D spectral density file or
          list of the five component files for directional spectra as
          [`spec`, `swdir`, `swdir2`, `swr1`, `swr2`].  There is no way to verify the
          component files for the historical directional spectra, so the order entered
          in the list is what is used. The history and realtime formats are
          automatically detected.
        - dirs (array): vector of directional bins for spectral reconstruction.
        - attrs (dict): additional global attributes.

    Returns:
        - dset (SpecDataset): spectra dataset object read from NDBC buoy file(s).

    """

    if isinstance(filename, str):
        filename = [filename]
    elif isinstance(filename, list):
        if not len(filename) == 5:
            raise ValueError(
                "filename argument for NDBC directional spectra must be list with 5 files [spden,swdir,swdir2,swr1,swr2]"
            )
    else:
        raise TypeError("filename argument must be string or list")

    # Get the spectra density
    df_spden = read_file(filename[0])

    if "Sep_Freq" in df_spden.columns:
        sep_freq = df_spden["Sep_Freq"].values
        df_spden.drop(columns=["Sep_Freq"], inplace=True)
    else:
        sep_freq = None

    times = df_spden.index
    freqs = df_spden.columns.astype("f")
    spshape = (len(times), len(freqs), 1)
    specdens = df_spden.values.reshape(spshape)

    if len(filename) == 1:
        dirs = [0.0]
    else:
        df_swdir = read_file(filename[1])
        df_swdir2 = read_file(filename[2])
        df_swr1 = read_file(filename[3])
        df_swr2 = read_file(filename[4])
        dirs = np.array(dirs)
        specdens = construct_spectra(
            specdens,
            df_swdir.values.reshape(spshape),
            df_swdir2.values.reshape(spshape),
            0.01
            * df_swr1.values.reshape(
                spshape
            ),  # these values are stored with a factor of 100
            0.01 * df_swr2.values.reshape(spshape),
            dirs,
        )
    coords = OrderedDict(
        ((attrs.TIMENAME, times), (attrs.FREQNAME, freqs), (attrs.DIRNAME, dirs))
    )
    dims = (attrs.TIMENAME, attrs.FREQNAME, attrs.DIRNAME)
    dset = xr.DataArray(
        data=specdens, coords=coords, dims=dims, name=attrs.SPECNAME
    ).to_dataset()
    if sep_freq is not None:
        sfreq = xr.DataArray(
            data=sep_freq,
            coords={attrs.TIMENAME: times},
            dims=(attrs.TIMENAME),
            name=attrs.SPECNAME,
        )
        dset[
            "Sep_Freq"
        ] = sfreq  # Add the NDBC defined separation frequency for realtime diagnostics
    dset = dset.sortby(
        "time", ascending=True
    )  # Realtime data is in reversed time order
    return dset
