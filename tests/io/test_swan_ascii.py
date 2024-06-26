from pathlib import Path
import os
import shutil
import numpy as np
import xarray as xr
import shutil
import pytest

from wavespectra.input.swan import read_swan, read_swans, read_hotswan, read_swanow
from wavespectra.core.attributes import attrs


FILENAME = Path(__file__).parent / "../sample_files/swanfile.spec"


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


def test_read_hotswan():
    """Test that swan hotstart file is read as a grid."""
    filename = FILENAME.parent / "swanhot.spec"
    ds = read_hotswan([filename])
    assert ds.lon.size > 1 and ds.lat.size > 1


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
    shutil.copy(str(FILENAME).replace(".spec", ".tab"), tabname)
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

    # ds = read_swans([specname])
    ds = read_swan(specname)
    assert {"wspd", "wdir", "dpt"}.issubset(dset)
    assert not {"wspd", "wdir", "dpt"}.issubset(ds)


def test_write_swanascii_no_latlon_specify(dset, tmpdir):
    ds = dset.drop_vars(("lon", "lat"))
    lons = [180.0]
    lats = [-30.0]
    filename = os.path.join(tmpdir, "spectra.swn")
    ds.spec.to_swan(filename, lons=lons, lats=lats)
    ds2 = read_swan(filename)
    assert sorted(ds2.lon.values) == sorted(lons)
    assert sorted(ds2.lat.values) == sorted(lats)


def test_write_swanascii_no_latlon_do_not_specify(dset, tmpdir):
    ds = dset.drop_vars(("lon", "lat"))
    filename = os.path.join(tmpdir, "spectra.swn")
    ds.spec.to_swan(filename)
    ds2 = read_swan(filename)
    assert list(ds2.lon.values) == list(ds2.lon.values * 0)
    assert list(ds2.lat.values) == list(ds2.lat.values * 0)


def test_write_swanascii_latlon_as_dims_and_no_site_dim(dset, tmpdir):
    ds = dset.isel(site=0).set_coords(("lon", "lat"))
    filename = os.path.join(tmpdir, "spectra.swn")
    ds.spec.to_swan(filename)
