from pathlib import Path
import numpy as np
import pytest

from wavespectra import read_datawell


FILES_DIR = Path(__file__).parent.parent / "sample_files/datawell"


@pytest.fixture(scope="module")
def filenames():
    yield sorted(FILES_DIR.glob("*.spt"))


def test_read_single_file_as_1d(filenames):
    dset = read_datawell(filenames[0], dd=None)
    assert dir not in dset.dims


def test_read_single_file_as_2d(filenames):
    dset = read_datawell(filenames[0], dd=5.0)
    assert np.diff(dset.dir.values).min() == 5


def test_read_single_file_no_latlon(filenames):
    dset = read_datawell(filenames[0], dd=5.0)
    assert "lon" not in dset.data_vars and "lat" not in dset.data_vars


def test_read_single_file_latlon(filenames):
    dset = read_datawell(filenames[0], dd=5.0, lon=3, lat=61)
    assert dset.lon == 3 and dset.lat == 61


def test_read_multiple_files(filenames):
    dset = read_datawell(filenames, dd=5.0)
    assert len(dset.time) == len(filenames)
