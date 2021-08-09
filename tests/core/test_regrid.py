import pytest
from pathlib import Path
import numpy as np

import wavespectra
from wavespectra import read_triaxys, read_ww3
from wavespectra.core.attributes import attrs
from wavespectra.core.utils import regrid_spec


FILES_DIR = Path(__file__).parent.parent / 'sample_files'


@pytest.fixture(scope="module")
def dset():
    _ds = read_triaxys(str(FILES_DIR / 'triaxys.DIRSPEC'))
    yield _ds


@pytest.fixture(scope="module")
def dset2():
    _ds = read_ww3(str(FILES_DIR / 'ww3file.nc'))
    yield _ds


def test_specarray_interp(dset):
    new_freq = np.linspace(0, 0.5, 25)
    new_dir = np.arange(0, 360, 5)
    dsi1 = regrid_spec(dset, dir=new_dir, freq=new_freq)
    dsi2 = dset.spec.interp(dir=new_dir, freq=new_freq)
    dsi1.equals(dsi2)


def test_specarray_interp_like(dset):
    new_freq = np.linspace(0, 0.5, 25)
    new_dir = np.arange(0, 360, 5)
    dsi1 = dset.spec.interp(dir=new_dir, freq=new_freq)
    dsi2 = dset.spec.interp_like(dsi1)
    dsi1.equals(dsi2)


def test_interp_freq(dset):
    new_freq = np.linspace(0, 0.5, 25)
    dsi = regrid_spec(dset, freq=new_freq)
    assert dset.spec.dir.equals(dsi.spec.dir)
    assert np.array_equal(dsi.spec.freq, new_freq)


def test_interp_dir(dset):
    new_dir = np.arange(0, 360, 5)
    dsi = regrid_spec(dset, dir=new_dir)
    assert dset.spec.freq.equals(dsi.spec.freq)
    assert np.array_equal(dsi.spec.dir, new_dir)


def test_maintain_m0(dset):
    new_freq = np.linspace(0, 0.5, 25)
    new_dir = np.arange(0, 360, 5)
    dsi = regrid_spec(dset, dir=new_dir, freq=new_freq)
    assert dset.spec.hs().values == pytest.approx(dsi.spec.hs().values)


def test_not_maintain_m0(dset):
    new_freq = np.linspace(0, 0.5, 25)
    new_dir = np.arange(0, 360, 5)
    dsi = regrid_spec(dset, dir=new_dir, freq=new_freq, maintain_m0=False)
    assert dset.spec.hs().values != pytest.approx(dsi.spec.hs().values)


def test_interp_lower_frequency(dset2):
    new_freq = np.linspace(0.01, 0.5, 25)
    dsi = regrid_spec(dset2, freq=new_freq)
    assert dsi.spec.oned().isel(site=0, time=0).isel(freq=0) > 0
    new_freq = np.linspace(0.0, 0.5, 25)
    dsi = regrid_spec(dset2, freq=new_freq)
    assert dsi.spec.oned().isel(site=0, time=0).isel(freq=0) == 0


def test_interp_higher_frequency(dset2):
    new_freq = np.linspace(0.01, 0.5, 25)
    dsi = regrid_spec(dset2, freq=new_freq)
    assert dsi.where(dsi.freq > dset2.freq.max(), drop=True).spec.hs().sum() == 0


def test_interp_lower_direction(dset2):
    dset = dset2.sortby("dir").isel(dir=slice(1, None))
    new_dir = np.arange(0.0, 360, 10)
    dsi = regrid_spec(dset, dir=new_dir)
    assert dsi.spec.dir.min() < dset.spec.dir.min()


def test_interp_upper_direction(dset2):
    dset = dset2.sortby("dir").isel(dir=slice(None, -1))
    new_dir = np.arange(0.0, 360, 10)
    dsi = regrid_spec(dset, dir=new_dir)
    assert dsi.spec.dir.max() > dset.spec.dir.max()
