"""Testing the SpecArray accessor."""
import os
import numpy as np
import xarray as xr
import pytest

from wavespectra import SpecArray
from wavespectra import _import_functions


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


@pytest.fixture(scope="module")
def dset():
    filename = os.path.join(FILES_DIR, "wavespectra.nc")
    _dset = xr.open_dataset(filename)
    yield _dset.efth


@pytest.fixture(scope="module")
def dset_full():
    filename = os.path.join(FILES_DIR, "wavespectra.nc")
    _dset = xr.open_dataset(filename)
    yield _dset


def test_accessor_attached(dset):
    assert hasattr(dset, "spec")


def test_repr(dset):
    assert "<SpecArray" in repr(dset.spec)


def test_interp_freq(dset):
    dset.spec._interp_freq(fint=0.2)
    with pytest.raises(ValueError):
        dset.spec._interp_freq(fint=5)


def test_split(dset):
    ds = dset.spec.split(fmin=0.1, fmax=0.2, dmin=0, dmax=15)
    ds.freq.min() == 0.1
    ds.freq.max() == 0.2
    ds.dir.min() == 0
    ds.dir.max() == 15
    with pytest.raises(ValueError):
        dset.spec.split(fmin=0.2, fmax=0.1)
    with pytest.raises(ValueError):
        dset.spec.split(dmin=15, dmax=0)


def test_hmax_defalt(dset):
    """The default factor k=1.86 is used if the dataset has no time coord."""
    ds = dset.isel(time=0, drop=True)
    hmax = ds.spec.hmax()
    assert hmax.values == pytest.approx(1.86 * ds.spec.hs().values)


def test_scale_by_hs(dset):
    ds = dset.spec.scale_by_hs(expr="2*hs")
    assert ds.spec.hs().values == pytest.approx(2 * dset.spec.hs().values)
    dset.spec.scale_by_hs(
        "2*hs", hs_min=1, hs_max=2, tp_min=5, tp_max=15, dpm_min=1, dpm_max=180
    )


def test_directional_methods_raise_on_oned_spec(dset):
    ds = dset.spec.oned()
    with pytest.raises(ValueError):
        ds.spec.momd()
    with pytest.raises(ValueError):
        ds.spec.dm()
    with pytest.raises(ValueError):
        ds.spec.dspr()
    with pytest.raises(ValueError):
        ds.spec.dpspr()
    with pytest.raises(ValueError):
        ds.spec.fdspr()
    with pytest.raises(ValueError):
        ds.spec.uss_x()
    with pytest.raises(ValueError):
        ds.spec.uss_y()


def test_crsd(dset):
    dset.spec.crsd()


def test_celerity(dset):
    shallow = dset.spec.celerity(depth=10)
    deep = dset.spec.celerity(depth=100)
    assert all(deep - shallow) > 0


def test_wavelen(dset):
    shallow = dset.spec.wavelen(depth=10)
    deep = dset.spec.wavelen(depth=100)
    assert all(deep - shallow) > 0


def test_stats(dset):
    hs1 = dset.spec.hs()
    hs2 = dset.spec.stats(["hs"]).hs
    assert hs1.values == pytest.approx(hs2.values)

    dset.spec.stats(["hs", "tp", "dpm"], names=["hs1", "tp1", "dpm1"])
    dset.spec.stats({"hs": {}, "crsd": {"theta": 90}})
    with pytest.raises(ValueError):
        dset.spec.stats("hs")
    with pytest.raises(ValueError):
        dset.spec.stats(["hs", "tp", "dpm"], names=["hs1", "tp1"])
    with pytest.raises(ValueError):
        dset.spec.stats(["stat_not_implemented"])
    with pytest.raises(ValueError):
        dset.spec.stats(["dd"])


def test_import_function():
    _import_functions("input", "read")
    _import_functions("dummy", "dummy2")


def test_one_frequency_bin(dset):
    ds = dset.isel(freq=[0])
    assert ds.spec.freq == dset.isel(freq=0).freq
    assert ds.spec.df.size == 1


def test_partition_interface(dset_full):
    dset = dset_full
    dset.spec.partition
    methods = ["ptm1", "ptm2", "ptm3", "ptm4", "ptm5"]
    for method in methods:
        assert method in dir(dset.spec.partition)
