from pathlib import Path
import os
import shutil
import numpy as np
import xarray as xr
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra.input.swan import read_swan, read_swans, read_hotswan, read_swanow
from wavespectra.core.attributes import attrs

FILES_DIR = Path(__file__).parent / "../sample_files"



@pytest.fixture(scope="module")
def dset():
    yield read_swan(FILENAME, as_site=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_read_swans(dset):
    ds = read_swans([FILENAME])
    assert dset.spec.hs().values == pytest.approx(
        ds.isel(site=[0]).spec.hs().values, rel=1e-2
    )
    ds = read_swans([FILENAME], int_freq=False)
    assert dset.spec.freq.equals(ds.spec.freq)
    ds = read_swans([FILENAME], int_dir=True)
    assert np.array_equal(ds.spec.dir.values, np.arange(0, 360, 10))
    ds = read_swans([FILENAME], ntimes=3)
    assert dset.time.size > ds.time.size
    ds = read_swans([FILENAME], ndays=1)
    assert dset.time.size > ds.time.size


def test_read_hotswan(dset):
    ds = read_hotswan([FILENAME])
    assert dset.spec.hs().values.ravel() == pytest.approx(
        ds.isel(lon=0, lat=0).spec.hs().values, rel=1e-2
    )


def test_read_swanow(dset):
    ds = read_swanow([FILENAME])
    assert dset.isel(site=0).spec.hs().values == pytest.approx(
        ds.isel(lon=0, lat=0, drop=True).spec.hs().values, rel=1e-2
    )


def test_read_swan_multiple_locations(dset, tmpdir):
    """Test reading swan file with more than one site."""
    filename = tmpdir / "swanfile.spec"
    ds = xr.concat([dset, dset, dset], dim="site")
    ds["site"] = [1, 2, 3]
    ds.spec.to_swan(filename)
    ds1 = ds.drop_vars(["wspd", "wdir", "dpt"])
    ds2 = read_swan(filename, as_site=True)
    assert ds1.equals(ds2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_read_swan_inconsistent_times(dset, tmpdir):
    """Test spectra is read without winds and depth if times are inconsistent."""
    specname = tmpdir / "swanfile.spec"
    tabname = tmpdir / "swanfile.tab"

    # Create inconsistent pairs
    shutil.copy(FILENAME.replace(".spec", ".tab"), tabname)
    dset.isel(time=[0, 1]).spec.to_swan(specname)

    ds = read_swans([specname])
    ds = read_swan(specname)
    assert {"wspd", "wdir", "dpt"}.issubset(dset)
    assert not {"wspd", "wdir", "dpt"}.issubset(ds)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_read_swan_bad_tabfile(dset, tmpdir):
    """Test spectra exception is raised for bad or nonsupported tabfile."""
    specname = tmpdir / "swanfile.spec"
    tabname = tmpdir / "swanfile.tab"

    # Tabfile is in fact a spectra file here which should fail
    shutil.copy(FILENAME, tabname)
    shutil.copy(FILENAME, specname)

    #ds = read_swans([specname])
    ds = read_swan(specname)
    assert {"wspd", "wdir", "dpt"}.issubset(dset)
    assert not {"wspd", "wdir", "dpt"}.issubset(ds)

