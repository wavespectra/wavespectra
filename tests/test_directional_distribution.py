"""Unit testing for stats methods in SpecArray."""
import os
import pytest
import datetime
import numpy as np
import xarray as xr

from wavespectra.directional import cartwright, bunney

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


@pytest.fixture(scope="module")
def dir():
    dirs = np.arange(0, 360, 15)
    _dir = xr.DataArray(dirs, coords={"dir": dirs}, dims=("dir",), name="dir")
    yield _dir


@pytest.fixture(scope="module")
def freq():
    freqs = np.arange(0, 0.5, 0.01)
    _freq = xr.DataArray(freqs, coords={"freq": freqs}, dims=("freq",), name="freq")
    yield _freq


def test_cartwright(dir):
    dm = 90
    dspr = 30
    gth = cartwright(dir, dm, dspr)


def test_bunney(dir, freq):
    dpm = 330
    dm = 350
    dspr = 30
    dpspr = 29
    fp = 0.1
    fm = 0.12
    bunney(dir=dir, freq=freq, dm=dm, dpm=dpm, dspr=dspr, dpspr=dpspr, fm=fm, fp=fp)
