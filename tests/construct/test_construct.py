"""Unit testing for stats methods in SpecArray."""
from pathlib import Path
import pytest
import numpy as np
import xarray as xr

from wavespectra.construct import construct_partition, partition_and_reconstruct
from wavespectra import read_swan


FILES_DIR = Path(__file__).parent.parent / "sample_files"


@pytest.fixture(scope="module")
def dset():
    filename = FILES_DIR / "swanfile.spec"
    _dset = read_swan(filename)
    yield _dset


@pytest.fixture(scope="module")
def freq():
    freqs = np.arange(0.03, 0.3, 0.001)
    _freq = xr.DataArray(freqs, coords={"freq": freqs}, dims=("freq",), name="freq")
    yield _freq


@pytest.fixture(scope="module")
def dir():
    dirs = np.arange(0, 360, 15)
    _dir = xr.DataArray(dirs, coords={"dir": dirs}, dims=("dir",), name="dir")
    yield _dir


def test_construct_partition(freq, dir):
    hs = 2
    fp = 0.1
    dm = 90
    dspr = 20
    freq_name = "jonswap"
    freq_kwargs = {"freq": freq, "hs": hs, "fp": fp}
    dir_name = "cartwright"
    dir_kwargs = {"dir": dir, "dm": dm, "dspr": dspr}

    dset = construct_partition(
        freq_name=freq_name,
        freq_kwargs=freq_kwargs,
        dir_name=dir_name,
        dir_kwargs=dir_kwargs,
    )

    assert float(dset.spec.hs()) == pytest.approx(hs, 1e5)
    assert float(dset.spec.fp()) == pytest.approx(fp, 1e5)
    assert float(dset.spec.dm()) == pytest.approx(dm, 1e5)
    assert dset.freq.values == pytest.approx(freq)
    assert dset.dir.values == pytest.approx(dir)


def test_partition_and_reconstruct_one_fit_all_partitions(dset):
    dsout = partition_and_reconstruct(
        dset,
        parts=4,
        freq_name="jonswap",
        dir_name="cartwright",
        method_combine="max",
    )


def test_partition_and_reconstruct_one_fit_per_partition(dset):
    dsout = partition_and_reconstruct(
        dset,
        parts=4,
        freq_name=["jonswap", "jonswap", "jonswap", "jonswap"],
        dir_name=["cartwright", "cartwright", "cartwright", "cartwright"],
        method_combine="max",
    )


def test_partition_and_reconstruct_inconsistent_number_of_partitions(dset):
    with pytest.raises(ValueError):
        dsout = partition_and_reconstruct(
            dset,
            parts=4,
            freq_name=["jonswap", "jonswap"],
            dir_name="cartwright",
            method_combine="max",
        )
    with pytest.raises(ValueError):
        dsout = partition_and_reconstruct(
            dset,
            parts=4,
            freq_name="jonswap",
            dir_name=["cartwright"],
            method_combine="max",
        )
