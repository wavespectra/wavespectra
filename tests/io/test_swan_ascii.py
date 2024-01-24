from pathlib import Path
import os
import xarray as xr
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra.input.swan import read_swan, read_swans, read_hotswan, read_swanow
from wavespectra.core.attributes import attrs

FILES_DIR = Path(__file__).parent / "../sample_files"



class TestSwan(object):
    """Test read swan ascii functions."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.filename = os.path.join(FILES_DIR, "swanfile.spec")
        self.filename_pathlib = FILES_DIR / "swanfile.spec"
        self.ds = read_swan(self.filename, as_site=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def test_read_swan(self):
        for filename in [self.filename, self.filename_pathlib]:
            ds = read_swan(filename)
            assert self.ds.isel(site=0).spec.hs().values == pytest.approx(
                ds.isel(lon=0, lat=0, drop=True).spec.hs().values, rel=1e-2
            )

    def test_read_swans(self):
        for filename in [self.filename, self.filename_pathlib]:
            ds = read_swans([filename])
            assert self.ds.spec.hs().values == pytest.approx(
                ds.isel(site=[0]).spec.hs().values, rel=1e-2
            )

    def test_read_hotswan(self):
        for filename in [self.filename, self.filename_pathlib]:
            ds = read_hotswan([filename])
            assert self.ds.spec.hs().values.ravel() == pytest.approx(
                ds.isel(lon=0, lat=0).spec.hs().values, rel=1e-2
            )

    def test_read_swanow(self):
        for filename in [self.filename, self.filename_pathlib]:
            ds = read_swanow([filename])
            assert self.ds.isel(site=0).spec.hs().values == pytest.approx(
                ds.isel(lon=0, lat=0, drop=True).spec.hs().values, rel=1e-2
            )

    def test_write_swanascii_no_latlon_specify(self):
        ds = self.ds.drop_vars(("lon", "lat"))
        lons = [180.0]
        lats = [-30.0]
        filename = os.path.join(self.tmp_dir, "spectra.swn")
        ds.spec.to_swan(filename, lons=lons, lats=lats)
        ds2 = read_swan(filename)
        assert sorted(ds2.lon.values) == sorted(lons)
        assert sorted(ds2.lat.values) == sorted(lats)

    def test_write_swanascii_no_latlon_do_not_specify(self):
        ds = self.ds.drop_vars(("lon", "lat"))
        filename = os.path.join(self.tmp_dir, "spectra.swn")
        ds.spec.to_swan(filename)
        ds2 = read_swan(filename)
        assert list(ds2.lon.values) == list(ds2.lon.values * 0)
        assert list(ds2.lat.values) == list(ds2.lat.values * 0)