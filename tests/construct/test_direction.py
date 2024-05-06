"""Unit testing for stats methods in SpecArray."""
import os
import pytest
import numpy as np
import xarray as xr

from wavespectra.construct.direction import cartwright, bunney

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


def test_cartwright_under_90(dir):
    dm = 180
    dspr = 50
    gth_cut = cartwright(dir, dm, dspr, under_90=True)
    gth_noncut = cartwright(dir, dm, dspr, under_90=False)
    above_90_ids = (dir < dm - 90) | (dir > dm + 90)
    assert gth_cut.where(above_90_ids).sum() == 0
    assert gth_noncut.where(above_90_ids).sum() > 0


def test_bunney(dir, freq):
    dpm = 330
    dm = 350
    dspr = 30
    dpspr = 29
    fp = 0.1
    fm = 0.12
    bunney(dir=dir, freq=freq, dm=dm, dpm=dpm, dspr=dspr, dpspr=dpspr, fm=fm, fp=fp)


def test_cartwright_dir_input_type(dir):
    """Test frequency input can also list, numpy or DataArray."""
    ds1 = cartwright(dir, 90, 30)
    ds2 = cartwright(dir.values, 90, 30)
    ds3 = cartwright(list(dir.values), 90, 30)
    assert ds1.equals(ds2)
    assert ds1.equals(ds3)


def test_bunney_dir_freq_input_type(dir, freq):
    """Test frequency input can also list, numpy or DataArray."""
    ds1 = bunney(dir=dir, freq=freq, dm=350, dpm=330, dspr=30, dpspr=29, fm=0.12, fp=0.1)
    ds2 = bunney(dir=dir, freq=freq.values, dm=350, dpm=330, dspr=30, dpspr=29, fm=0.12, fp=0.1)
    ds3 = bunney(dir=dir, freq=list(freq.values), dm=350, dpm=330, dspr=30, dpspr=29, fm=0.12, fp=0.1)
    ds4 = bunney(dir=dir.values, freq=freq, dm=350, dpm=330, dspr=30, dpspr=29, fm=0.12, fp=0.1)
    ds5 = bunney(dir=list(dir.values), freq=freq, dm=350, dpm=330, dspr=30, dpspr=29, fm=0.12, fp=0.1)
    assert ds1.equals(ds2)
    assert ds1.equals(ds3)
    assert ds1.equals(ds4)
    assert ds1.equals(ds5)
