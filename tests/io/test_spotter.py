from pathlib import Path
import numpy as np
import pytest

from wavespectra import read_spotter


FILES_DIR = Path(__file__).parent.parent / "sample_files"


@pytest.fixture(scope="module")
def csv_file():
    yield sorted(FILES_DIR.glob("spotter*.csv"))[0]


@pytest.fixture(scope="module")
def json_file():
    yield sorted(FILES_DIR.glob("spotter*.json"))[0]


def test_read_single_file_as_1d(csv_file):
    dset = read_spotter(csv_file, dd=None)
    assert dir not in dset.dims


def test_read_single_file_as_2d(json_file):
    dset = read_spotter(json_file, dd=5.0)
    assert np.diff(dset.dir.values).min() == 5


def test_read_multiple_files(csv_file):
    dset1 = read_spotter(csv_file, dd=5.0)
    dset2 = read_spotter([csv_file, csv_file], dd=5.0)
    assert dset2.time.size == 2 * dset1.time.size


@pytest.mark.parametrize(
    "stat, kwargs",
    [
        ("hs", {}),
        ("tp", {"smooth": False}),
        ("dpm", {}),
    ],
)
def test_stats(stat, kwargs):
    """Assert that stat calculated from wavespectra matches the one read from spotter."""
    filenames = sorted(FILES_DIR.glob("spotter*"))
    for filename in filenames:
        dset = read_spotter(filename)
        param_wavespectra = getattr(dset.spec, stat)(**kwargs).values
        param_spotter = dset[stat].values
        assert param_wavespectra == pytest.approx(param_spotter, rel=1e-1)
