"""Unit testing for stats methods in SpecArray."""
import os
import pytest
import datetime
import numpy as np
import xarray as xr

from wavespectra.directional_distribution import cartwright, bunney

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


@pytest.fixture(scope="module")
def dir():
    dirs = np.arange(0, 360, 15)
    _dir = xr.DataArray(dirs, coords={"dir": dirs}, dims=("dir",), name="dir")
    yield _dir


def test_cartwright(dir):
    dm = 90
    dspr = 30
    gth = cartwright(dir, dm, dspr)


def test_bunney(dir):
    dpm = 330
    dm = 350
    fp = 0.1
    fm = 0.12
    bunney(dir, dpm, dm, fp, fm)
